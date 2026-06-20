"""
Qwen2-VL 行人跟踪 - LoRA 微调训练脚本

方案 B (温和稳定): 回归原生检测格式 + 后处理跟踪
  Stage 1: 单帧检测 SFT (LoRA r=8, lr=2e-5, 3 epoch)
           - 使用原生 Qwen2-VL 检测格式, 利用模型已有能力
           - 目标: 让模型学会在 MOT17 上输出准确的行人 bbox
  Stage 2: GRPO RL 精调 (IoU奖励 + ID一致性奖励, 优化跟踪精度)

使用方式:
  python model_method/train.py --stage 1
  python model_method/train.py --stage 2
  python model_method/train.py --allstage
  python model_method/train.py --stage 1 --quick
"""

import os
import sys
import argparse
from pathlib import Path

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
)
from peft import get_peft_model, prepare_model_for_kbit_training, PeftModel
MODEL_PATH = PROJECT_ROOT / "Qwen"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 数据文件映射:
# Stage 1 (SFT) 使用单帧检测数据 (原生 Qwen2-VL 格式, 利用模型已有检测能力)
# Stage 2 (RL) 使用双帧跟踪数据, 用于奖励计算
DATA_FILE_MAP = {
    1: "mot17_sft_stage1.jsonl",  # SFT: 单帧检测 (原生格式)
    2: "mot17_sft_stage2.jsonl",  # RL: 双帧跟踪数据
}

# 修复 Windows GBK 编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# 各阶段训练配置
# ============================================================

def get_stage_config(stage, output_dir):
    """获取各阶段的训练配置

    优化策略 (RTX4060 8GB + Windows):
    - gradient_checkpointing 必开 (省显存)
    - use_reentrant=False (LoRA兼容 + 更快)
    - save_total_limit=2 (减少磁盘占用)
    - Stage 1: lr=2e-5, 3 epoch (单帧检测, 利用模型已有能力, 适度微调)
    """
    common_kwargs = dict(
        per_device_train_batch_size=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        max_grad_norm=0.3,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=False,
    )

    configs = {
        1: TrainingArguments(
            output_dir=str(output_dir / "stage1"),
            num_train_epochs=3,           # 3 epoch, 单帧检测微调
            gradient_accumulation_steps=8,
            learning_rate=2e-5,           # 适中学习率, 避免灾难性遗忘
            **common_kwargs,
        ),
    }
    return configs[stage]


# ============================================================
# 模型加载
# ============================================================

def load_model_and_processor(model_path, use_qlora=False):
    """加载 Qwen2-VL 模型和处理器"""
    print(f"  加载模型: {model_path}")

    if use_qlora:
        try:
            import bitsandbytes
            use_qlora = True
        except ImportError:
            print("  ⚠️ bitsandbytes 未安装, 自动切换到 fp16 LoRA")
            use_qlora = False

    print(f"  QLoRA: {use_qlora}")

    if use_qlora:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=torch.float16,
            device_map="auto",
        )

    processor = AutoProcessor.from_pretrained(str(model_path))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params / 1e9:.2f}B")

    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  GPU 显存占用: {vram_mb:.0f} MB")

    return model, processor


# ============================================================
# Stage 1: SFT 检测训练
# ============================================================

def train_sft(args):
    """Stage 1: 单帧检测 SFT (LoRA r=8, 原生 Qwen2-VL 检测格式)"""

    print("=" * 60)
    print("Stage 1: 单帧检测 SFT (原生格式)")
    print("=" * 60)

    # 1. 加载模型
    model, processor = load_model_and_processor(MODEL_PATH, use_qlora=args.qlora)

    # gradient checkpointing 兼容性
    model.enable_input_require_grads()
    model.config.use_cache = False

    # 2. 注入 LoRA (r=8, 极小, 保留检测能力)
    if args.resume_from:
        print(f"  从已有 LoRA 权重继续训练: {args.resume_from}")
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
    else:
        from model_method.lora_config import get_tracking_lora_config
        lora_config = get_tracking_lora_config()
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # 3. 加载数据 (单帧检测数据, 原生格式)
    from model_method.dataset import TrackingSFTDataset

    data_path = DATA_DIR / DATA_FILE_MAP[1]
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        print(f"  请先运行: python model_method/build_sft_data.py --stages 1")
        return

    # 单帧检测可用更高分辨率 (相比双帧的 1568, 单帧用 6272, 4倍分辨率)
    # 更高分辨率有助于检测小目标行人
    max_pixels = 28 * 28 * 8  # 6272

    train_dataset = TrackingSFTDataset(
        data_path=str(data_path),
        processor=processor,
        max_seq_length=2048,   # 单帧序列更短, 2048 足够
        max_pixels=max_pixels,
    )

    collate_fn = train_dataset.collate_fn
    if args.quick:
        quick_n = min(20, len(train_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, range(quick_n))
        print(f"  ⚡ Quick 模式: 只使用 {quick_n} 个样本")

    # 4. 辅助损失函数 (坐标加权, 每4步计算一次)
    from model_method.losses import CoordinateWeightedLoss

    aux_loss_fn = CoordinateWeightedLoss(
        processor.tokenizer,
        coord_weight=2.0,
    )
    print(f"  辅助损失: CoordinateWeightedLoss (coord_weight=2.0, 每4步)")

    # 5. 训练配置
    training_args = get_stage_config(1, OUTPUT_DIR)
    if args.epochs:
        training_args.num_train_epochs = args.epochs
    if args.lr:
        training_args.learning_rate = args.lr

    if args.quick:
        training_args.num_train_epochs = 1
        training_args.gradient_accumulation_steps = 2
        training_args.logging_steps = 1
        training_args.save_steps = 50
        print(f"  ⚡ Quick 模式: 1 epoch, grad_accum=2, 每步打印loss")

    # 6. 创建 Trainer
    from model_method.tracking_trainer import TrackingTrainer

    trainer = TrackingTrainer(
        aux_loss_fn=aux_loss_fn,
        iou_every_n_steps=0,
        aux_every_n_steps=4,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    # 7. 开始训练
    print(f"\n  开始训练 Stage 1 (单帧检测 SFT)...")
    print(f"  训练样本数: {len(train_dataset)}")
    print(f"  训练轮次: {training_args.num_train_epochs}")
    print(f"  学习率: {training_args.learning_rate}")
    print(f"  批次大小: {training_args.per_device_train_batch_size}")
    print(f"  梯度累积: {training_args.gradient_accumulation_steps}")
    print(f"  图像分辨率 (max_pixels): {max_pixels}")
    print()

    trainer.train()

    # 8. 保存
    save_path = str(OUTPUT_DIR / "stage1" / "final")
    trainer.save_model(save_path)
    print(f"\n  ✅ LoRA 权重已保存: {save_path}")

    processor.save_pretrained(save_path)
    print(f"  ✅ Processor 已保存: {save_path}")

    # 9. 释放显存
    del model, trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print(f"  ✅ 显存已释放")


# ============================================================
# Stage 2: GRPO RL 训练
# ============================================================

def train_rl(args):
    """Stage 2: GRPO RL 精调"""

    print("=" * 60)
    print("Stage 2: GRPO RL 精调")
    print("=" * 60)

    import json
    from datasets import Dataset as HFDataset

    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError:
        print("❌ trl 未安装, Stage 2 需要 GRPO 训练")
        print("  请运行: pip install trl>=0.12.0")
        sys.exit(1)

    from model_method.rl_reward import tracking_reward_fn, parse_bboxes_from_text, parse_tracks_from_text

    # 1. 确定上一阶段权重路径
    sft_path = args.resume_from
    if sft_path is None:
        sft_path = str(OUTPUT_DIR / "stage1" / "final")
    if not Path(sft_path).exists():
        print(f"❌ SFT 权重不存在: {sft_path}")
        print("  请先运行: python model_method/train.py --stage 1")
        sys.exit(1)

    # 2. 加载模型
    model, processor = load_model_and_processor(MODEL_PATH, use_qlora=args.qlora)

    print(f"  加载 SFT LoRA: {sft_path}")
    model = PeftModel.from_pretrained(model, sft_path, is_trainable=True)

    model.enable_input_require_grads()
    model.config.use_cache = False

    model.print_trainable_parameters()

    # 3. 加载数据
    data_path = DATA_DIR / DATA_FILE_MAP[2]
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        sys.exit(1)

    processor.image_processor.max_pixels = 28 * 28 * 2  # 1568
    processor.image_processor.min_pixels = 28 * 28  # 784

    # 构建 RL 数据集
    samples = []
    with open(str(data_path), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            messages = sample['messages']

            prompt_messages = [m for m in messages if m['role'] != 'assistant']

            gt_bboxes = []
            gt_ids = []
            for m in messages:
                if m['role'] == 'assistant':
                    for c in m.get('content', []):
                        if c.get('type') == 'text':
                            gt_bboxes.extend(parse_bboxes_from_text(c['text']))
                            tracks = parse_tracks_from_text(c['text'])
                            gt_ids.extend(list(tracks.keys()))

            samples.append({
                'prompt': prompt_messages,
                'gt_bboxes': gt_bboxes,
                'gt_ids': gt_ids,
            })

    print(f"  训练样本数: {len(samples)}")

    num_generations = 2
    if args.quick:
        quick_n = min(10, len(samples))
        samples = samples[:quick_n]
        num_generations = 2
        print(f"  ⚡ Quick 模式: 只使用 {quick_n} 个样本, G=2")

    hf_dataset = HFDataset.from_dict({
        'prompt': [s['prompt'] for s in samples],
        'gt_bboxes': [s['gt_bboxes'] for s in samples],
        'gt_ids': [s['gt_ids'] for s in samples],
    })

    # 4. GRPO 配置
    lr = args.lr if args.lr else 1e-5
    epochs = args.epochs if args.epochs else 1

    grpo_config = GRPOConfig(
        output_dir=str(OUTPUT_DIR / "stage2"),
        num_generations=num_generations,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=1,
        save_steps=100,
        max_completion_length=256,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        max_grad_norm=0.3,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    if args.quick:
        grpo_config.num_train_epochs = 1
        grpo_config.gradient_accumulation_steps = 2
        print(f"  ⚡ Quick 模式: 1 epoch, G={num_generations}")

    # 5. 创建 Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor.tokenizer,
        reward_funcs=[tracking_reward_fn],
        args=grpo_config,
        train_dataset=hf_dataset,
    )

    # 6. 开始训练
    print(f"\n  开始训练 Stage 2 (GRPO RL)...")
    print(f"  训练样本数: {len(hf_dataset)}")
    print(f"  生成数量 G: {num_generations}")
    print(f"  学习率: {lr}")
    print(f"  训练轮次: {grpo_config.num_train_epochs}")
    print()

    trainer.train()

    # 7. 保存
    save_path = str(OUTPUT_DIR / "stage2" / "final")
    trainer.save_model(save_path)
    print(f"\n  ✅ RL LoRA 权重已保存: {save_path}")

    processor.save_pretrained(save_path)
    print(f"  ✅ Processor 已保存: {save_path}")

    # 8. 释放显存
    del model, trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print(f"  ✅ 显存已释放")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL 行人跟踪 LoRA 微调训练")
    parser.add_argument(
        '--stage', type=int, default=1, choices=[1, 2],
        help='训练阶段 (1=单帧检测SFT, 2=GRPO RL精调)'
    )
    parser.add_argument(
        '--qlora', action='store_true', default=False,
        help='使用 QLoRA (4bit 量化, 需要 bitsandbytes)'
    )
    parser.add_argument(
        '--no-qlora', action='store_true',
        help='不使用 QLoRA (fp16 训练, 默认)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='覆盖默认训练轮次'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='覆盖默认学习率'
    )
    parser.add_argument(
        '--quick', action='store_true', default=False,
        help='快速验证模式: 少量样本, 1 epoch (~10分钟)'
    )
    parser.add_argument(
        '--allstage', action='store_true', default=False,
        help='一次性训练全部2个阶段 (Stage 1→2)'
    )
    parser.add_argument(
        '--resume-from', type=str, default=None,
        help='从指定 LoRA 权重路径继续训练'
    )
    args = parser.parse_args()

    if args.no_qlora:
        args.qlora = False

    print("=" * 60)
    print("Qwen2-VL 行人跟踪 - LoRA 微调训练")
    print("=" * 60)
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  训练阶段: {'全部 (1→2)' if args.allstage else f'Stage {args.stage}'}")
    print(f"  QLoRA: {args.qlora}")
    print(f"  Quick: {args.quick}")
    print()

    # 检查模型
    if not MODEL_PATH.exists():
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        sys.exit(1)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --allstage: 依次训练 2 个阶段
    if args.allstage:
        print(f"\n{'='*60}")
        print("开始 Stage 1 训练 (单帧检测 SFT)")
        print(f"{'='*60}\n")
        train_sft(args)
        print(f"\n{'='*60}")
        print("Stage 1 训练完成")
        print(f"{'='*60}\n")

        print(f"\n{'='*60}")
        print("开始 Stage 2 训练 (GRPO RL)")
        print(f"{'='*60}\n")
        train_rl(args)
        print(f"\n{'='*60}")
        print("Stage 2 训练完成")
        print(f"{'='*60}\n")

        print("✅ 全部 2 个阶段训练完成!")
        print(f"  Stage 1 权重 (SFT检测): {OUTPUT_DIR / 'stage1' / 'final'}")
        print(f"  Stage 2 权重 (RL):      {OUTPUT_DIR / 'stage2' / 'final'}")
    elif args.stage == 1:
        train_sft(args)
    elif args.stage == 2:
        train_rl(args)


if __name__ == '__main__':
    main()
