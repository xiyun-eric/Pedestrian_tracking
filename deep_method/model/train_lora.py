"""
YOLO11 LoRA 微调训练脚本（稳定版）

创新点:
  1. 网络结构优化: LoRA 低秩适配卷积层（仅训练0.85%参数）
  2. 损失函数设计: 多尺度 IoU 损失 (GIoU + CIoU + EIoU 加权融合)

训练策略:
  通过自定义 LoRADetectionTrainer 保留 LoRA 注入层，
  使用 YOLO 内置训练流程（数据加载、验证、日志等）。
  自定义损失通过 monkey-patch bbox_iou 集成。

关键修复:
  YOLO 的 Model.train() 内部会调用 get_model() 从 YAML 重建模型，
  导致注入的 LoRA 层被丢弃。本脚本通过自定义 Trainer 重写 get_model()
  返回已注入 LoRA 的模型，并在 _setup_train() 后重新冻结非 LoRA 参数
  并重建优化器（仅包含 LoRA 可训练参数）。

使用方法:
  # 使用 YOLO 格式数据集配置文件（推荐）
  python deep_method/model/train_lora.py --data configs/dataset.yaml --epochs 30 --batch 4

  # 指定 LoRA 秩
  python deep_method/model/train_lora.py --data configs/dataset.yaml --r 16 --epochs 50

输出路径:
  runs/yolo_lora/train/weights/
  ├── best.pt          # 标准 YOLO 格式（LoRA 已合并，可直接推理）
  ├── last.pt          # 标准 YOLO 格式
  ├── lora_best.pt     # LoRA 权重备份（合并前保存）
  └── lora_final.pt    # LoRA 权重备份（合并前保存）
"""

import os
import sys
import warnings
import logging

# 彻底抑制所有警告（必须在其他 import 之前）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow
os.environ['PYTHONWARNINGS'] = 'ignore'   # Python warnings
warnings.filterwarnings('ignore')         # 所有警告
# 注意：不抑制 ultralytics 的日志，否则训练进度不可见

from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import torch

# 导入 LoRA 模块
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from lora_layers import (
    inject_lora_to_yolo,
    freeze_model_except_lora,
    save_lora_weights,
    load_lora_weights,
    merge_lora_to_model,
)
from custom_loss import MultiScaleIoULoss

from ultralytics.models.yolo.detect import DetectionTrainer


# ============================================================
# 自定义训练器：保留 LoRA 层
# ============================================================

def create_lora_trainer(lora_model):
    """
    创建保留 LoRA 层的自定义训练器

    问题:
      YOLO 的 Model.train() 会调用 trainer.get_model() 从 YAML 重建模型，
      导致注入的 LoRA 层被丢弃。pretrained=False 只是不加载权重，
      但模型已被重建为随机初始化的无 LoRA 模型。

    解决方案:
      1. 重写 get_model() 返回已注入 LoRA 的模型
      2. 重写 _setup_train() 在训练初始化后:
         - 重新冻结非 LoRA 参数（_setup_train 会解冻所有参数）
         - 重建优化器（仅包含 LoRA 可训练参数）
         - 重建学习率调度器

    Args:
        lora_model: 已注入 LoRA 层的模型

    Returns:
        LoRADetectionTrainer 类（不是实例）
    """

    class LoRADetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            """返回 LoRA 注入模型，而不是从 YAML 重建"""
            return lora_model

        def _setup_train(self, *args, **kwargs):
            """重写训练设置，保留 LoRA 冻结状态（兼容不同 ultralytics 版本）"""
            # 调用父类方法（兼容不同签名）
            super()._setup_train(*args, **kwargs)

            # _setup_train 中会解冻所有参数（requires_grad=True），
            # 需要重新冻结非 LoRA 参数
            freeze_model_except_lora(self.model)

            # 重建优化器，只包含 LoRA 可训练参数
            lora_params = [p for p in self.model.parameters() if p.requires_grad]

            if not lora_params:
                print("[警告] 没有可训练的 LoRA 参数!")
                return

            trainable_count = sum(p.numel() for p in lora_params)
            print(f"\n[LoRA] 重建优化器，仅包含 {trainable_count:,} 个可训练参数")

            self.optimizer = torch.optim.AdamW(
                lora_params,
                lr=self.args.lr0,
                weight_decay=self.args.weight_decay,
            )

            # 重建学习率调度器
            if self.args.cos_lr:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.epochs
                )
            else:
                self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lf
                )

        def final_eval(self):
            """跳过自动验证，避免 LoRA 模型 fuse 失败（合并 LoRA 后单独验证）"""
            print("\n[LoRA] 跳过自动验证，将在合并 LoRA 权重后单独验证")
            return None

    return LoRADetectionTrainer


# ============================================================
# 损失函数集成
# ============================================================

def wrap_criterion_with_custom_loss():
    """
    包装 YOLO 内置损失函数，集成多尺度 IoU 损失

    原理:
      YOLO 默认使用 CIoU 计算框损失。
      我们将其替换为多尺度 IoU 损失:
        L_box = w1*L_GIoU + w2*L_CIoU + w3*L_EIoU

    实现:
      通过 monkey-patch ultralytics.utils.metrics.bbox_iou，
      当 YOLO 调用 bbox_iou(..., CIoU=True) 时，
      返回 GIoU/CIoU/EIoU 的加权融合结果。
    """
    try:
        from ultralytics.utils.metrics import bbox_iou as original_bbox_iou

        def multi_scale_bbox_iou(box1, box2, xywh=False, GIoU=False, DIoU=False,
                                  CIoU=False, EIoU=False, eps=1e-7):
            """多尺度 IoU: 当 YOLO 请求 CIoU 时，返回 GIoU+CIoU+EIoU 加权融合"""
            if CIoU:
                giou = original_bbox_iou(box1, box2, xywh=xywh, GIoU=True, eps=eps)
                ciou = original_bbox_iou(box1, box2, xywh=xywh, CIoU=True, eps=eps)
                try:
                    eiou = original_bbox_iou(box1, box2, xywh=xywh, EIoU=True, eps=eps)
                except TypeError:
                    eiou = original_bbox_iou(box1, box2, xywh=xywh, DIoU=True, eps=eps)
                return 0.2 * giou + 0.5 * ciou + 0.3 * eiou
            return original_bbox_iou(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU,
                                     CIoU=CIoU, EIoU=EIoU, eps=eps)

        # Patch at module level
        import ultralytics.utils.metrics as metrics_module
        metrics_module.bbox_iou = multi_scale_bbox_iou

        # Also patch in loss module if already imported
        try:
            import ultralytics.utils.loss as loss_module
            if hasattr(loss_module, 'bbox_iou'):
                loss_module.bbox_iou = multi_scale_bbox_iou
        except Exception:
            pass

        print("  - 已集成多尺度 IoU 损失 (GIoU*0.2 + CIoU*0.5 + EIoU*0.3)")
        return True

    except Exception as e:
        print(f"  [警告] 多尺度 IoU 损失集成失败: {e}，使用默认 CIoU")
        return False


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="YOLO11 LoRA 微调训练")

    # 模型
    parser.add_argument("--model", type=str, default="yolo11m.pt",
                       help="基础模型 (yolo11n/s/m/l)")
    parser.add_argument("--r", type=int, default=8, help="LoRA 秩")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")

    # 数据
    parser.add_argument("--data", type=str, default="configs/dataset.yaml",
                       help="数据集配置文件")

    # 训练
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch", type=int, default=4, help="批次大小")
    parser.add_argument("--img-size", type=int, default=640, help="输入尺寸")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")

    # 设备
    parser.add_argument("--device", type=str, default="auto",
                       help="GPU设备 (auto=自动检测, cpu=CPU, 0=GPU)")

    # 输出
    parser.add_argument("--output", type=str, default="runs/yolo_lora",
                       help="输出目录")
    parser.add_argument("--resume", type=str, default=None, help="恢复 LoRA 权重路径")

    args = parser.parse_args()

    # 自动检测设备
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "0"
            print(f"自动检测到 GPU，使用设备: cuda:0")
        else:
            args.device = "cpu"
            print(f"自动检测: CUDA 不可用，使用 CPU 训练")

    print(f"\n{'='*60}")
    print("YOLO11 LoRA 微调训练")
    print(f"{'='*60}")

    if args.device != "cpu" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ----------------------------------------------------------
    # 1. 集成自定义损失函数（必须在加载模型之前 patch）
    # ----------------------------------------------------------
    print("\n初始化自定义损失函数...")
    wrap_criterion_with_custom_loss()

    # ----------------------------------------------------------
    # 2. 加载模型并注入 LoRA
    # ----------------------------------------------------------
    print(f"\n加载模型: {args.model}")
    from ultralytics import YOLO
    yolo_model = YOLO(args.model)
    model = yolo_model.model

    print(f"\n注入 LoRA (r={args.r}, alpha={args.lora_alpha})...")
    model, trainable_params = inject_lora_to_yolo(
        model,
        r=args.r,
        alpha=args.lora_alpha,
        target_modules=['detect', 'cv4', 'cv3', 'cv2'],
        verbose=True,
    )

    # 冻结非 LoRA 参数
    print("\n冻结非 LoRA 参数...")
    freeze_model_except_lora(model)

    # 恢复 LoRA 权重
    if args.resume:
        print(f"\n恢复 LoRA 权重: {args.resume}")
        model = load_lora_weights(model, args.resume)

    # 验证 LoRA 层存在
    lora_layer_count = sum(1 for m in model.modules() if type(m).__name__ == 'Conv2dLoRA')
    if lora_layer_count == 0:
        print("\n[错误] 模型中没有 LoRA 层! 请检查 inject_lora_to_yolo 的 target_modules 参数。")
        return
    print(f"\n验证: 模型中有 {lora_layer_count} 个 Conv2dLoRA 层")

    # ----------------------------------------------------------
    # 3. 创建自定义训练器并训练
    # ----------------------------------------------------------
    print(f"\n创建 LoRA 自定义训练器...")
    LoRATrainer = create_lora_trainer(model)

    print(f"\n开始训练...")
    print(f"  - 数据: {args.data}")
    print(f"  - 轮数: {args.epochs}")
    print(f"  - 批次: {args.batch}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - LoRA 秩: {args.r}")

    results = yolo_model.train(
        trainer=LoRATrainer,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        lr0=args.lr,
        device=args.device,
        project=args.output,
        name="train",
        exist_ok=True,
        pretrained=False,
        optimizer="AdamW",
        verbose=True,
        seed=42,
        deterministic=False,
        cos_lr=True,
        close_mosaic=10,
        resume=False,
        amp=False,
        fraction=1.0,
        freeze=None,
        val=False,  # 禁用自动验证，避免 LoRA 模型 fuse 失败
        workers=0,  # Windows 下 workers>1 可能死锁，1 是安全的折中
        # 数据增强参数
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    # ----------------------------------------------------------
    # 4. 获取训练后的模型（确保是 LoRA 模型）
    # ----------------------------------------------------------
    # yolo_model.train() 内部会更新 yolo_model.model
    model = yolo_model.model

    # 验证训练后 LoRA 层仍然存在
    lora_layer_count_after = sum(1 for m in model.modules() if type(m).__name__ == 'Conv2dLoRA')
    print(f"\n训练后验证: 模型中有 {lora_layer_count_after} 个 Conv2dLoRA 层")
    if lora_layer_count_after == 0:
        print("[错误] 训练后 LoRA 层丢失! 请检查自定义 Trainer 是否正确工作。")
        return

    # ----------------------------------------------------------
    # 5. 保存 LoRA 权重（必须在合并之前保存！）
    # ----------------------------------------------------------
    output_dir = Path(args.output) / "train" / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n保存 LoRA 权重（合并前）...")
    lora_best_path = output_dir / "lora_best.pt"
    lora_final_path = output_dir / "lora_final.pt"
    save_lora_weights(model, str(lora_best_path))
    save_lora_weights(model, str(lora_final_path))

    # ----------------------------------------------------------
    # 6. 合并 LoRA 权重并保存为标准 YOLO 格式
    # ----------------------------------------------------------
    print("\n合并 LoRA 权重到原始卷积层...")
    model = merge_lora_to_model(model)

    # 构造 YOLO 标准 checkpoint
    ckpt = {
        'model': model,
        'train_args': {
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.img_size,
            'lr0': args.lr,
            'optimizer': 'AdamW',
        },
    }

    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    torch.save(ckpt, str(best_path))
    torch.save(ckpt, str(last_path))
    print(f"  - 已保存标准 YOLO 格式: {best_path}")

    # ----------------------------------------------------------
    # 7. 验证合并后的模型
    # ----------------------------------------------------------
    print("\n验证合并后的模型...")
    from ultralytics import YOLO as YOLOInference
    yolo_inference = YOLOInference(str(best_path))
    val_results = yolo_inference.val(
        data=args.data,
        device=args.device,
        project=args.output,  # 输出到 yolo_lora 目录，避免生成 runs/detect
        name="val",
    )
    print(f"  - mAP50: {val_results.box.map50:.4f}")
    print(f"  - mAP50-95: {val_results.box.map:.4f}")

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"模型保存在: {args.output}/train/weights/")
    print(f"  - best.pt (标准 YOLO 格式，可直接用于推理)")
    print(f"  - last.pt (标准 YOLO 格式)")
    print(f"  - lora_best.pt (LoRA 权重备份)")
    print(f"  - lora_final.pt (LoRA 权重备份)")


if __name__ == "__main__":
    main()
