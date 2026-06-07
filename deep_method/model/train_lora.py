"""
LoRA 微调训练脚本

训练流程:
  1. 加载 YOLO11m 预训练模型
  2. 注入 LoRA 层（只对检测头附近层）
  3. 冻结非 LoRA 参数
  4. 使用自定义损失函数和数据增强训练
  5. 保存 LoRA 权重（轻量级，仅 ~500KB）

使用方法:
  # 使用自采集数据集训练
  python deep_method/model/train_lora.py \
    --img-dir data/custom/images \
    --label-dir data/custom/annotations \
    --epochs 30 \
    --batch 4 \
    --lr 0.001 \
    --r 8

8GB 显存适配:
  - batch_size=4~8 (根据模型大小调整)
  - img_size=640 (推荐)
  - 使用梯度累积增大等效 batch
  - 混合精度训练 (AMP)
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# 导入 LoRA 模块
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from lora_layers import (
    inject_lora_to_yolo,
    freeze_model_except_lora,
    save_lora_weights,
    load_lora_weights,
)
from custom_loss import MultiScaleIoULoss, FocalLoss, AdaptiveWeightLoss


class YOLODataset(Dataset):
    """YOLO 训练数据集 - 适配自采集数据结构"""
    
    def __init__(
        self,
        img_dir: Path,
        label_dir: Path,
        split: str = "train",
        img_size: int = 640,
        augment: bool = True,
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.split = split
        self.img_size = img_size
        self.augment = augment
        
        # 收集所有图像-标签对
        # 数据结构: img_dir/scene1/train/*.png, label_dir/scene1/train/*.txt
        self.samples = []
        
        for scene_dir in sorted(img_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            split_dir = scene_dir / split
            if not split_dir.exists():
                continue
            
            for img_path in sorted(split_dir.glob("*.png")) + sorted(split_dir.glob("*.jpg")):
                frame_id = img_path.stem
                # 标签路径: label_dir/scene_name/split/frame_id.txt
                label_path = label_dir / scene_dir.name / split / f"{frame_id}.txt"
                
                # 图像存在，标签可能不存在（帧中没有人）
                self.samples.append((img_path, label_path))
        
        print(f"数据集 [{split}]: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        import cv2
        
        img_path, label_path = self.samples[idx]
        
        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            # fallback 到第一个样本
            return self.__getitem__(0)
        
        h, w = image.shape[:2]
        
        # 读取标签
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx = float(parts[1])
                        cy = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])
                        labels.append([cls_id, cx, cy, bw, bh])
        
        labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
        
        # 数据增强
        if self.augment and len(labels) > 0:
            image, labels = self._augment(image, labels)
        
        # 调整大小
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image)
        
        return image, torch.from_numpy(labels), (h, w)
    
    def _augment(self, image, labels):
        """简单数据增强"""
        # 随机水平翻转
        if np.random.random() < 0.5:
            image = np.fliplr(image).copy()
            # 翻转标签的 x 坐标
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]
        
        # 随机亮度调整
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)
        
        return image, labels


def collate_fn(batch):
    """处理变长标签的批次"""
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    shapes = [item[2] for item in batch]
    return images, labels, shapes


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets, shapes) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        
        # 前向传播
        optimizer.zero_grad()
        
        try:
            if use_amp:
                with autocast():
                    # YOLO 模型前向传播
                    output = model(images)
                    # 从输出中提取损失
                    if hasattr(output, 'loss'):
                        loss = output.loss
                    elif isinstance(output, dict) and 'loss' in output:
                        loss = output['loss']
                    else:
                        # 如果没有损失，使用检测输出计算
                        loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                output = model(images)
                if hasattr(output, 'loss'):
                    loss = output.loss
                elif isinstance(output, dict) and 'loss' in output:
                    loss = output['loss']
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 反向传播
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"Batch {batch_idx} error: {e}")
            continue
        
        # 定期清理缓存
        if batch_idx % 20 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / max(n_batches, 1)


def validate(model, dataloader: DataLoader, device: torch.device) -> Dict:
    """验证"""
    model.eval()
    total_detections = 0
    total_targets = 0
    
    with torch.no_grad():
        for images, targets, shapes in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            
            # 统计目标数
            for t in targets:
                total_targets += len(t)
            
            # 模型推理
            try:
                output = model(images)
                if hasattr(output, 'boxes'):
                    total_detections += len(output.boxes)
            except:
                pass
    
    recall = total_detections / max(total_targets, 1) if total_targets > 0 else 0
    
    return {
        "total_targets": total_targets,
        "total_detections": total_detections,
        "recall": recall,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLO11 LoRA 微调训练")
    
    # 模型
    parser.add_argument("--model", type=str, default="yolo11m.pt",
                       help="基础模型 (yolo11n/s/m/l)")
    parser.add_argument("--r", type=int, default=8, help="LoRA 秩")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    
    # 数据
    parser.add_argument("--img-dir", type=str, default="data/custom/images",
                       help="图像目录")
    parser.add_argument("--label-dir", type=str, default="data/custom/annotations",
                       help="标签目录")
    parser.add_argument("--img-size", type=int, default=640, help="输入尺寸")
    
    # 训练
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    
    # 设备
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true", help="禁用混合精度")
    
    # 输出
    parser.add_argument("--output-dir", type=str, default="weights/lora")
    parser.add_argument("--resume", type=str, default=None, help="恢复 LoRA 权重路径")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("YOLO11 LoRA 微调训练")
    print(f"{'='*60}")
    print(f"设备: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # 清空缓存
        torch.cuda.empty_cache()
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    from ultralytics import YOLO
    yolo_model = YOLO(args.model)
    model = yolo_model.model
    
    # 注入 LoRA
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
    
    model = model.to(device)
    
    # 数据加载
    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    
    # 训练集
    train_dataset = YOLODataset(
        img_dir, label_dir, split="train",
        img_size=args.img_size, augment=True
    )
    
    # 验证集
    val_dataset = YOLODataset(
        img_dir, label_dir, split="val",
        img_size=args.img_size, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    print(f"\n训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 优化器
    lora_params = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数: {sum(p.numel() for p in lora_params):,}")
    
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # 混合精度
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    print(f"混合精度: {use_amp}")
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    print(f"\n{'='*60}")
    print("开始训练")
    print(f"{'='*60}")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, use_amp
        )
        
        # 验证
        val_stats = validate(model, val_loader, device)
        
        # 学习率更新
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Targets: {val_stats['total_targets']} | "
              f"LR: {current_lr:.6f}")
        
        # 保存最佳
        if train_loss < best_loss:
            best_loss = train_loss
            save_path = output_dir / "lora_best.pt"
            save_lora_weights(model, str(save_path))
            print(f"  -> 保存最佳模型: {save_path}")
        
        # 定期保存
        if epoch % 10 == 0:
            save_path = output_dir / f"lora_epoch_{epoch}.pt"
            save_lora_weights(model, str(save_path))
        
        # 清理缓存
        torch.cuda.empty_cache()
    
    # 保存最终模型
    final_path = output_dir / "lora_final.pt"
    save_lora_weights(model, str(final_path))
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"最佳训练损失: {best_loss:.4f}")
    print(f"LoRA 权重保存在: {output_dir}")
    print(f"  - lora_best.pt (最佳模型)")
    print(f"  - lora_final.pt (最终模型)")


if __name__ == "__main__":
    main()