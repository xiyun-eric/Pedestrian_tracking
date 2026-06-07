"""
使用 Ultralytics 内置训练功能进行 YOLO11 微调

更简单可靠的训练方式，直接使用 YOLO 的 train() 方法

使用方法:
  python deep_method/model/train_yolo.py --epochs 30 --batch 4
"""

import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description="YOLO11 微调训练")
    
    # 模型
    parser.add_argument("--model", type=str, default="yolo11m.pt",
                       help="基础模型")
    
    # 数据
    parser.add_argument("--data", type=str, default="configs/dataset.yaml",
                       help="数据集配置文件")
    
    # 训练
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch", type=int, default=4, help="批次大小")
    parser.add_argument("--img-size", type=int, default=640, help="输入尺寸")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    
    # 设备
    parser.add_argument("--device", type=str, default="auto", help="GPU设备 (auto=自动检测, cpu=CPU, 0=GPU)")
    
    # 输出
    parser.add_argument("--output", type=str, default="weights/yolo_custom",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 自动检测设备
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "0"
            print(f"自动检测到 GPU，使用设备: cuda:0")
        else:
            args.device = "cpu"
            print(f"自动检测: CUDA 不可用，使用 CPU 训练")
    
    # 设置设备
    print(f"\n{'='*60}")
    print("YOLO11 微调训练")
    print(f"{'='*60}")
    
    if args.device != "cpu" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("使用 CPU 训练")
    
    # 检查数据配置
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"错误: 数据配置文件不存在: {data_path}")
        print("请先创建 configs/dataset.yaml")
        return
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    from ultralytics import YOLO
    model = YOLO(args.model)
    
    # 开始训练
    print(f"\n开始训练...")
    print(f"  - 数据: {args.data}")
    print(f"  - 轮数: {args.epochs}")
    print(f"  - 批次: {args.batch}")
    print(f"  - 图像尺寸: {args.img_size}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 设备: {args.device}")
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.img_size,
        lr0=args.lr,
        device=args.device,
        project=args.output,
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        verbose=True,
        seed=42,
        deterministic=False,
        single_cls=False,  # 多类别
        rect=False,
        cos_lr=True,  # 余弦学习率调度
        close_mosaic=10,  # 最后10轮关闭 mosaic 增强
        resume=False,
        amp=False,  # 禁用混合精度（避免 AMP 检查下载超时）
        fraction=1.0,
        profile=False,
        freeze=None,  # 不冻结任何层
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
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"模型保存在: {args.output}/train/")
    print(f"  - best.pt (最佳模型)")
    print(f"  - last.pt (最终模型)")


if __name__ == "__main__":
    main()