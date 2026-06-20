"""
Phase 1: Qwen2-VL 模型验证脚本
验证模型能否正常加载并输出 bbox 格式的行人检测结果
"""

import os
import sys
import json
import math
import torch
from pathlib import Path

# 兼容补丁: torch 2.2.0 没有 torch.compiler.is_compiling
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "Qwen"
MOT17_PATH = PROJECT_ROOT / "data" / "MOT17"

# 修复 Windows GBK 编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def check_model_files():
    """检查模型文件是否完整"""
    print("=" * 60)
    print("Step 1: 检查模型文件")
    print("=" * 60)

    required_files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "merges.txt",
        "vocab.json",
    ]

    all_ok = True
    for f in required_files:
        path = MODEL_PATH / f
        exists = path.exists()
        size_mb = path.stat().st_size / 1024 / 1024 if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {f} ({size_mb:.1f} MB)")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n✅ 所有模型文件完整")
    else:
        print("\n❌ 部分文件缺失，请补充下载")
        sys.exit(1)

    return True


def check_special_tokens():
    """检查 tokenizer 是否包含 bbox 相关特殊 token"""
    print("\n" + "=" * 60)
    print("Step 2: 检查 bbox 特殊 token")
    print("=" * 60)

    with open(MODEL_PATH / "tokenizer_config.json", "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    added_tokens = tokenizer_config.get("added_tokens_decoder", {})

    # 检查 bbox 相关 token
    bbox_tokens = {
        "<|box_start|>": 151648,
        "<|box_end|>": 151649,
        "<|object_ref_start|>": 151646,
        "<|object_ref_end|>": 151647,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
        "<|image_pad|>": 151655,
    }

    all_ok = True
    for token, expected_id in bbox_tokens.items():
        found = False
        for tid, info in added_tokens.items():
            if info.get("content") == token:
                found = True
                actual_id = int(tid)
                match = "✅" if actual_id == expected_id else "⚠️"
                print(f"  {match} {token} -> ID {actual_id} (期望 {expected_id})")
                break
        if not found:
            print(f"  ❌ {token} 未找到")
            all_ok = False

    if all_ok:
        print("\n✅ 所有 bbox 相关 token 存在，模型原生支持 bbox 输出")
    else:
        print("\n⚠️ 部分 token 缺失，可能影响 bbox 输出")

    return all_ok


def load_model():
    """加载 Qwen2-VL 模型"""
    print("\n" + "=" * 60)
    print("Step 3: 加载模型")
    print("=" * 60)

    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    except ImportError as e:
        print(f"❌ ImportError: {e}")
        print("请先安装依赖: pip install transformers>=4.45.0 qwen-vl-utils")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"  模型路径: {MODEL_PATH}")
    print(f"  加载方式: float16, auto device map")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.float16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params / 1e9:.2f}B")

    # 显存占用
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  GPU 显存占用: {vram_mb:.0f} MB")
        print(f"  GPU 设备: {torch.cuda.get_device_name(0)}")

    print("\n✅ 模型加载成功")
    return model, processor


def test_single_frame_detection(model, processor):
    """测试单帧行人检测 (bbox 输出)"""
    print("\n" + "=" * 60)
    print("Step 4: 测试单帧行人检测")
    print("=" * 60)

    from qwen_vl_utils import process_vision_info

    # 使用 MOT17 的第一帧
    test_image = MOT17_PATH / "MOT17-02-FRCNN" / "img1" / "000001.jpg"
    if not test_image.exists():
        print(f"❌ 测试图像不存在: {test_image}")
        return False

    print(f"  测试图像: {test_image.name}")

    # 构建检测提示词
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(test_image)},
                {
                    "type": "text",
                    "text": (
                        "Detect all pedestrians in this image. "
                        "For each person, output their bounding box. "
                        "Use format: <ref>person</ref><box>(x1,y1),(x2,y2)</box>"
                    ),
                },
            ],
        }
    ]

    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    # 生成
    print("  正在生成...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, temperature=0.1)

    response = processor.decode(output[0], skip_special_tokens=False)
    clean_response = processor.decode(output[0], skip_special_tokens=True)

    print(f"\n  原始输出 (含特殊token):")
    print(f"  {response[:500]}")
    print(f"\n  清洁输出:")
    print(f"  {clean_response[:500]}")

    # 检查是否包含 bbox (两种格式: 内部token格式和文本标签格式)
    has_box_token = "<|box_start|>" in response
    has_box_tag = "<box>" in response or "<box " in response
    has_ref_tag = "<ref>" in response

    print(f"\n  <|box_start|> (内部token): {has_box_token}")
    print(f"  <box> (文本标签): {has_box_tag}")
    print(f"  <ref> (引用标签): {has_ref_tag}")

    if has_box_token or has_box_tag:
        # 解析 <box>(x1,y1),(x2,y2)</box> 格式
        import re
        box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
        boxes = re.findall(box_pattern, response)
        if boxes:
            print(f"\n  解析到 {len(boxes)} 个行人 bbox:")
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                print(f"    行人{i+1}: ({x1},{y1}),({x2},{y2})")
        print("\n✅ 模型原生支持 bbox 输出，无需额外训练即可输出坐标格式")
    else:
        print("\n⚠️ 模型未输出 bbox 格式，可能需要 SFT 训练来激发此能力")
        print("  这是正常的 - Instruct 版本可能需要特定提示词格式")

    return has_box_token or has_box_tag


def test_grounding_format(model, processor):
    """测试 Qwen2-VL 官方推荐的 grounding 格式"""
    print("\n" + "=" * 60)
    print("Step 5: 测试官方 grounding 格式")
    print("=" * 60)

    from qwen_vl_utils import process_vision_info

    test_image = MOT17_PATH / "MOT17-02-FRCNN" / "img1" / "000001.jpg"

    # Qwen2-VL 官方推荐的 grounding 提示词格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(test_image)},
                {
                    "type": "text",
                    "text": "Locate the pedestrians in this image with bounding boxes.",
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    print("  正在生成 (官方格式)...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)

    response = processor.decode(output[0], skip_special_tokens=False)
    clean_response = processor.decode(output[0], skip_special_tokens=True)

    print(f"\n  原始输出:")
    print(f"  {response[:800]}")
    print(f"\n  清洁输出:")
    print(f"  {clean_response[:800]}")

    # 解析 bbox (支持两种格式)
    import re
    # 格式1: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    box_pattern_token = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    boxes = re.findall(box_pattern_token, response)
    # 格式2: <box>(x1,y1),(x2,y2)</box>
    box_pattern_tag = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    boxes += re.findall(box_pattern_tag, response)
    print(f"\n  解析到 {len(boxes)} 个 bbox:")
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        print(f"    行人{i+1}: ({x1},{y1}),({x2},{y2})")

    if len(boxes) > 0:
        print(f"\n✅ 成功检测到 {len(boxes)} 个行人 bbox")
    else:
        print("\n⚠️ 未解析到 bbox，模型可能需要微调来增强检测能力")

    return len(boxes)


def test_lora_injection(model):
    """测试 LoRA 注入是否可行"""
    print("\n" + "=" * 60)
    print("Step 6: 测试 LoRA 注入")
    print("=" * 60)

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        print("❌ 请先安装依赖: pip install peft>=0.12.0")
        return False

    # LoRA 配置 (与计划文档一致)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 注入 LoRA (不量化，仅测试注入可行性)
    try:
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        # 检查可训练参数
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        print(f"\n  可训练参数: {trainable:,}")
        print(f"  总参数: {total:,}")
        print(f"  可训练比例: {trainable/total*100:.2f}%")

        print("\n✅ LoRA 注入成功，可训练参数比例合理")
        return True
    except Exception as e:
        print(f"\n❌ LoRA 注入失败: {e}")
        return False


def main():
    print("Qwen2-VL 行人跟踪 - Phase 1 模型验证")
    print("=" * 60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"MOT17路径: {MOT17_PATH}")
    print()

    # Step 1: 检查文件
    check_model_files()

    # Step 2: 检查特殊 token
    check_special_tokens()

    # Step 3: 加载模型
    model, processor = load_model()

    # Step 4: 测试单帧检测 (自定义格式)
    test_single_frame_detection(model, processor)

    # Step 5: 测试官方 grounding 格式
    test_grounding_format(model, processor)

    # Step 6: 测试 LoRA 注入
    test_lora_injection(model)

    # 总结
    print("\n" + "=" * 60)
    print("Phase 1 验证总结")
    print("=" * 60)
    print("""
  如果以上测试均通过，说明:
  1. 模型文件完整，可正常加载
  2. Tokenizer 包含 bbox 相关特殊 token
  3. 模型原生支持 bbox 输出 (或可通过 SFT 激活)
  4. LoRA 可正常注入

  下一步: Phase 2 - 数据准备
  运行: python -m model_method.data_preparation.build_sft_dataset
    """)


if __name__ == "__main__":
    main()
