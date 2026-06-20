"""
OSNet LoRA 微调训练脚本（ReID 行人重识别）

创新点:
  1. 网络结构优化: LoRA 低秩适配 OSNet 的 Conv2d(1x1) 和 Linear 层
  2. 损失函数设计: 联合损失 = 标签平滑交叉熵 + Batch-Hard Triplet Loss + Center Loss

数据来源:
  - MOT17 数据集: 从 GT 标注中提取行人裁剪图像，按身份ID组织
  - 也支持自定义 ReID 数据集（Market-1501 格式）

训练流程:
  1. 解析 MOT17 GT，提取行人裁剪图像并按身份组织
  2. 构建 ReID 数据集（每个身份一个文件夹）
  3. 加载 OSNet 预训练权重，注入 LoRA 层
  4. 使用联合 ReID 损失训练
  5. 合并 LoRA 权重，保存完整模型

使用方法:
  # 使用 MOT17 数据集训练
  python deep_method/model/train_osnet_lora.py --data data/MOT17 --epochs 30

  # 指定 LoRA 秩和权重
  python deep_method/model/train_osnet_lora.py --data data/MOT17 --r 8 --weights osnet_x1_0_msmt17.pth

  # 使用自定义 ReID 数据集（Market-1501 格式: dataset/identity_id/img.jpg）
  python deep_method/model/train_osnet_lora.py --data /path/to/reid_dataset --format market1501

输出路径（与 YOLO 训练区分）:
  runs/reid/osnet_lora/
  ├── best.pth          # 最佳完整模型（合并 LoRA 后）
  ├── last.pth          # 最终完整模型
  ├── lora_best.pt      # LoRA 权重备份
  ├── lora_last.pt      # LoRA 权重备份
  └── train_log.json    # 训练日志
"""

import sys
import os
from pathlib import Path

# 抑制警告（必须在其他 import 之前）
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TensorFlow 警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning)  # 抑制 torchreid Cython 警告

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import argparse
import os
import json
import time
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# 导入 LoRA 模块
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from osnet_lora import (
    inject_lora_to_osnet,
    freeze_model_except_lora,
    save_lora_weights,
    load_lora_weights,
    save_merged_model,
)
from reid_loss import CombinedReIDLoss


# ============================================================
# 数据集
# ============================================================

class MOT17ReIDDataset(Dataset):
    """
    从 MOT17 数据集构建 ReID 训练数据

    流程:
    1. 读取 gt.txt，获取每个身份在各帧的边界框
    2. 从对应帧图像中裁剪行人区域
    3. 按身份组织数据，每个身份有多个裁剪图像
    """

    def __init__(
        self,
        mot_dirs: List[Path],
        transform: Optional[T.Compose] = None,
        min_samples: int = 2,
        crop_padding: float = 0.1,
    ):
        """
        Args:
            mot_dirs: MOT17 序列目录列表 (如 [MOT17-04-FRCNN, ...])
            transform: 图像变换
            min_samples: 每个身份最少样本数（少于此数的身份被过滤）
            crop_padding: 裁剪时边界扩展比例
        """
        self.transform = transform or T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.min_samples = min_samples
        self.crop_padding = crop_padding

        # 收集所有裁剪数据
        self.samples = []  # [(image_path, bbox, pid_global), ...]
        self.pid_to_global = {}  # (seq_name, local_pid) -> global_pid
        self.num_identities = 0

        for mot_dir in mot_dirs:
            self._load_mot_sequence(mot_dir)

        # 过滤样本数不足的身份
        self._filter_identities()

        print(f"[数据集] 共 {len(self.samples)} 个裁剪, {self.num_identities} 个身份")

    def _load_mot_sequence(self, mot_dir: Path):
        """加载单个 MOT17 序列"""
        gt_path = mot_dir / 'gt' / 'gt.txt'
        if not gt_path.exists():
            print(f"[警告] GT 文件不存在: {gt_path}")
            return

        # 读取 seqinfo
        seqinfo_path = mot_dir / 'seqinfo.ini'
        img_dir_name = 'img1'
        if seqinfo_path.exists():
            with open(seqinfo_path, 'r') as f:
                for line in f:
                    if line.startswith('imDir'):
                        img_dir_name = line.split('=')[1].strip()

        img_dir = mot_dir / img_dir_name

        # 读取 GT
        gt_data = defaultdict(list)  # pid -> [(frame, x, y, w, h), ...]
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue
                frame_id = int(parts[0])
                pid = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                conf = float(parts[6])

                # 跳过忽略的标注（pid < 1 或 conf == 0）
                if pid < 1 or conf == 0:
                    continue

                gt_data[pid].append((frame_id, x, y, w, h))

        # 为每个身份分配全局 ID
        seq_name = mot_dir.name
        for local_pid in sorted(gt_data.keys()):
            key = (seq_name, local_pid)
            if key not in self.pid_to_global:
                self.pid_to_global[key] = self.num_identities
                self.num_identities += 1

            global_pid = self.pid_to_global[key]
            for frame_id, x, y, w, h in gt_data[local_pid]:
                # 查找图像文件
                img_path = img_dir / f"{frame_id:06d}.jpg"
                if not img_path.exists():
                    img_path = img_dir / f"{frame_id:06d}.png"
                if not img_path.exists():
                    continue

                self.samples.append((str(img_path), (x, y, w, h), global_pid))

    def _filter_identities(self):
        """过滤样本数不足的身份，重新映射 ID"""
        # 统计每个身份的样本数
        pid_counts = defaultdict(int)
        for _, _, pid in self.samples:
            pid_counts[pid] += 1

        # 过滤
        valid_pids = {pid for pid, count in pid_counts.items() if count >= self.min_samples}

        # 重新映射
        pid_map = {}
        new_pid = 0
        for old_pid in sorted(valid_pids):
            pid_map[old_pid] = new_pid
            new_pid += 1

        self.samples = [
            (img_path, bbox, pid_map[pid])
            for img_path, bbox, pid in self.samples
            if pid in valid_pids
        ]
        self.num_identities = new_pid

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, pid = self.samples[idx]

        # 读取图像
        image = Image.open(img_path).convert('RGB')

        # 裁剪行人区域（带 padding）
        x, y, w, h = bbox
        img_w, img_h = image.size
        pad = self.crop_padding

        x1 = max(0, int(x - w * pad))
        y1 = max(0, int(y - h * pad))
        x2 = min(img_w, int(x + w * (1 + pad)))
        y2 = min(img_h, int(y + h * (1 + pad)))

        crop = image.crop((x1, y1, x2, y2))

        # 变换
        if self.transform:
            crop = self.transform(crop)

        return crop, pid


class Market1501Dataset(Dataset):
    """
    Market-1501 格式的 ReID 数据集

    目录结构:
      dataset/
      ├── 0001/  (身份1)
      │   ├── 0001_c1_001.jpg
      │   └── ...
      ├── 0002/  (身份2)
      └── ...
    """

    def __init__(
        self,
        root: Path,
        transform: Optional[T.Compose] = None,
    ):
        self.transform = transform or T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.samples = []
        self.num_identities = 0

        # 遍历身份目录
        id_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        for pid, id_dir in enumerate(id_dirs):
            for img_path in id_dir.glob('*.jpg'):
                self.samples.append((str(img_path), pid))
            for img_path in id_dir.glob('*.png'):
                self.samples.append((str(img_path), pid))

        self.num_identities = len(id_dirs)
        print(f"[数据集] Market1501 格式: {len(self.samples)} 图像, {self.num_identities} 身份")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pid = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, pid


# ============================================================
# 训练工具函数
# ============================================================

class FeatureExtractor(nn.Module):
    """
    OSNet 特征提取器包装

    训练时: 返回 (logits, features)
    推理时: 只返回 features
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # OSNet forward: 训练模式返回 logits，eval 模式返回 features
        if self.model.training:
            # 训练时需要 logits 和 features
            # OSNet 的 forward 在训练时返回 logits
            logits = self.model(x)
            # 提取 features（倒数第二层）
            features = self._extract_features(x)
            return logits, features
        else:
            features = self.model(x)
            return features

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取 512 维特征（跳过 classifier）"""
        # 遍历模型层，在 classifier 之前截断
        features = x
        for name, module in self.model.named_children():
            if name == 'classifier':
                break
            features = module(features)

        # 全局平均池化
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)

        return features


def build_osnet_model(
    weights_path: Optional[str] = None,
    num_classes: int = 1000,
    device: str = 'cpu',
    replace_classifier: bool = False,
) -> nn.Module:
    """构建 OSNet 模型并加载预训练权重

    Args:
        weights_path: 预训练权重路径
        num_classes: 数据集类别数（仅在 replace_classifier=True 时使用）
        device: 设备
        replace_classifier: 是否替换 classifier（默认 False，保持预训练的 4101 类）
    """
    import torchreid

    # 先用预训练权重的类别数创建模型
    pretrained_num_classes = num_classes
    if weights_path and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if 'classifier.weight' in state_dict:
            pretrained_num_classes = state_dict['classifier.weight'].shape[0]

    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=pretrained_num_classes,
        pretrained=False,
    )

    # 加载权重
    if weights_path and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=True)
        print(f"[模型] 已加载预训练权重: {weights_path} (pretrained_classes={pretrained_num_classes})")

        # 仅在显式要求时替换 classifier
        if replace_classifier and pretrained_num_classes != num_classes:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            print(f"[模型] 替换 classifier: {pretrained_num_classes} -> {num_classes}")
        else:
            print(f"[模型] 保持 classifier: {pretrained_num_classes} 类（不替换，避免过拟合）")
    else:
        print(f"[模型] 未加载预训练权重，使用随机初始化")

    return model.to(device), pretrained_num_classes


def evaluate_reid(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedReIDLoss,
    device: str,
    features_capture: dict,
) -> Dict[str, float]:
    """评估 ReID 模型"""
    model.eval()
    total_loss = 0
    total_cls = 0
    total_tri = 0
    total_center = 0
    num_batches = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  验证", ncols=80, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播（hook 自动捕获 features）
            features_capture.clear()
            logits = model(images)

            features = features_capture.get('features', logits)
            features = F.normalize(features, p=2, dim=1)

            # 计算损失
            losses = criterion(logits, features, labels)
            total_loss += losses['total'].item()
            total_cls += losses['cls'].item()
            total_tri += losses['triplet'].item()
            total_center += losses['center'].item()
            num_batches += 1

    model.train()

    if num_batches == 0:
        return {'total': 0, 'cls': 0, 'triplet': 0, 'center': 0}

    return {
        'total': total_loss / num_batches,
        'cls': total_cls / num_batches,
        'triplet': total_tri / num_batches,
        'center': total_center / num_batches,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="OSNet LoRA 微调训练 (ReID)")

    # 数据
    parser.add_argument("--data", type=str, default="data/MOT17",
                       help="数据集路径 (MOT17目录 或 Market-1501格式目录)")
    parser.add_argument("--format", type=str, default="mot17",
                       choices=["mot17", "market1501"],
                       help="数据集格式")
    parser.add_argument("--min-samples", type=int, default=2,
                       help="每个身份最少样本数")

    # 模型
    parser.add_argument("--weights", type=str, default="osnet_x1_0_msmt17.pth",
                       help="OSNet 预训练权重路径")
    parser.add_argument("--r", type=int, default=8, help="LoRA 秩")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--target-layers", type=str, default=None,
                       help="LoRA 目标层 (逗号分隔, 默认: layer3,layer4,classifier)")

    # 训练
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")

    # 损失函数权重
    parser.add_argument("--w-cls", type=float, default=1.0, help="分类损失权重")
    parser.add_argument("--w-tri", type=float, default=1.0, help="三元组损失权重")
    parser.add_argument("--w-center", type=float, default=0.0005, help="中心损失权重")
    parser.add_argument("--triplet-margin", type=float, default=0.3, help="三元组间隔")
    parser.add_argument("--label-smooth", type=float, default=0.1, help="标签平滑系数")

    # 设备
    parser.add_argument("--device", type=str, default="auto",
                       help="设备 (auto/cpu/0)")

    # 输出（与 YOLO 训练区分）
    parser.add_argument("--output", type=str, default="runs/reid/osnet_lora",
                       help="输出目录")
    parser.add_argument("--resume-lora", type=str, default=None,
                       help="恢复 LoRA 权重路径")

    args = parser.parse_args()

    # 自动检测设备
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "0"
            print(f"自动检测到 GPU，使用设备: cuda:0")
        else:
            args.device = "cpu"
            print(f"自动检测: CUDA 不可用，使用 CPU 训练")

    device = f"cuda:{args.device}" if args.device.isdigit() else args.device

    print(f"\n{'='*60}")
    print("OSNet LoRA 微调训练 (ReID 行人重识别)")
    print(f"{'='*60}")

    if device.startswith('cuda') and torch.cuda.is_available():
        gpu_id = int(args.device) if args.device.isdigit() else 0
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"显存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB")

    # ----------------------------------------------------------
    # 1. 构建数据集
    # ----------------------------------------------------------
    print(f"\n构建数据集...")

    # 验证变换（无数据增强）
    val_transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.format == 'mot17':
        # 查找 MOT17 子目录
        data_path = Path(args.data)
        mot_dirs = []
        for subdir in sorted(data_path.iterdir()):
            if subdir.is_dir() and (subdir / 'gt' / 'gt.txt').exists():
                mot_dirs.append(subdir)

        # 也检查 temp 目录
        temp_path = _project_root / 'temp'
        if temp_path.exists():
            for subdir in sorted(temp_path.iterdir()):
                if subdir.is_dir() and (subdir / 'gt' / 'gt.txt').exists():
                    mot_dirs.append(subdir)

        if not mot_dirs:
            print(f"错误: 未找到 MOT17 数据 (在 {data_path} 和 {temp_path})")
            return

        print(f"  找到 {len(mot_dirs)} 个 MOT17 序列:")
        for d in mot_dirs:
            print(f"    - {d.name}")

        # 训练集（带数据增强）
        train_transform = T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])

        # 使用所有数据训练（MOT17 数据量有限）
        full_dataset = MOT17ReIDDataset(
            mot_dirs=mot_dirs,
            transform=train_transform,
            min_samples=args.min_samples,
        )

        num_classes = full_dataset.num_identities
        train_dataset = full_dataset
        val_dataset = MOT17ReIDDataset(
            mot_dirs=mot_dirs,
            transform=val_transform,
            min_samples=args.min_samples,
        )

    else:  # market1501
        data_path = Path(args.data)
        train_dir = data_path / 'bounding_box_train'
        val_dir = data_path / 'bounding_box_test'

        if not train_dir.exists():
            train_dir = data_path

        train_dataset = Market1501Dataset(train_dir)
        num_classes = train_dataset.num_identities

        if val_dir.exists():
            val_dataset = Market1501Dataset(val_dir, transform=val_transform)
        else:
            val_dataset = train_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.startswith('cuda'),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith('cuda'),
    )

    print(f"  训练集: {len(train_dataset)} 图像, {num_classes} 身份")
    print(f"  验证集: {len(val_dataset)} 图像")

    # ----------------------------------------------------------
    # 2. 加载模型并注入 LoRA
    # ----------------------------------------------------------
    print(f"\n加载 OSNet 模型...")
    model, pretrained_num_classes = build_osnet_model(
        weights_path=args.weights,
        num_classes=num_classes,
        device=device,
        replace_classifier=False,  # 不替换 classifier，保持 4101 类
    )

    # 注入 LoRA
    target_layers = args.target_layers.split(',') if args.target_layers else None
    print(f"\n注入 LoRA (r={args.r}, alpha={args.lora_alpha})...")
    model, trainable_params = inject_lora_to_osnet(
        model,
        r=args.r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_layers=target_layers,
        verbose=True,
    )

    # 确保 LoRA 层在正确设备上
    model = model.to(device)

    # 冻结非 LoRA 参数
    print("\n冻结非 LoRA 参数...")
    freeze_model_except_lora(model)

    # 恢复 LoRA 权重
    if args.resume_lora:
        print(f"\n恢复 LoRA 权重: {args.resume_lora}")
        model = load_lora_weights(model, args.resume_lora)

    # 设置为训练模式
    model.train()

    # ----------------------------------------------------------
    # 3. 构建损失函数和优化器
    # ----------------------------------------------------------
    print(f"\n构建损失函数...")
    # 由于 classifier 保持预训练的 4101 类且不参与训练，不使用分类损失
    # 只使用 TripletLoss + CenterLoss 进行度量学习
    # CenterLoss 使用实际身份数（而非 4101），避免大量无用中心浪费内存
    criterion = CombinedReIDLoss(
        num_classes=num_classes,  # 使用实际身份数，而非 4101
        feat_dim=512,
        w_cls=0.0,  # 不使用分类损失（classifier 保持预训练权重）
        w_tri=args.w_tri,
        w_center=args.w_center,
        triplet_margin=args.triplet_margin,
        label_smooth_epsilon=args.label_smooth,
    )
    print(f"  - CrossEntropyLabelSmooth: 禁用（classifier 保持预训练权重）")
    print(f"  - TripletLoss (margin={args.triplet_margin}, batch_hard)")
    print(f"  - CenterLoss (lr=0.5)")
    print(f"  - 权重: cls=0.0, tri={args.w_tri}, center={args.w_center}")

    # 只优化 LoRA 参数 + center_loss centers（classifier 不参与训练）
    lora_params = []

    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
        # classifier 不加入优化器：w_cls=0.0 无有效梯度，参与训练反而损害泛化

    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': args.lr, 'weight_decay': 1e-4},
        {'params': criterion.center_loss.parameters(), 'lr': args.lr * 0.1},
    ])

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ----------------------------------------------------------
    # 4. 训练循环
    # ----------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n开始训练...")
    print(f"  - 数据: {args.data}")
    print(f"  - 轮数: {args.epochs}")
    print(f"  - 批次: {args.batch}")
    print(f"  - 学习率: {args.lr}")
    print(f"  - LoRA 秩: {args.r}")
    print(f"  - 输出: {args.output}")

    best_loss = float('inf')
    train_log = []

    # 在训练循环外注册一次 hook（避免每个 batch 注册/移除的开销）
    features_capture = {}
    hook_handle = _register_feature_hook(model, features_capture)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_cls = 0
        epoch_tri = 0
        epoch_center = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=100)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 前向传播（hook 自动捕获 features）
            features_capture.clear()
            logits = model(images)

            # 获取 hook 捕获的特征并归一化
            features = features_capture.get('features', logits)
            features = F.normalize(features, p=2, dim=1)

            # 计算损失
            losses = criterion(logits, features, labels)
            loss = losses['total']

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cls += losses['cls'].item()
            epoch_tri += losses['triplet'].item()
            epoch_center += losses['center'].item()
            num_batches += 1

            # 更新进度条信息
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls': f'{losses["cls"].item():.3f}',
                'tri': f'{losses["triplet"].item():.3f}',
            })

        scheduler.step()

        # 计算平均损失
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_cls = epoch_cls / max(num_batches, 1)
        avg_tri = epoch_tri / max(num_batches, 1)
        avg_center = epoch_center / max(num_batches, 1)

        # 验证
        val_metrics = evaluate_reid(model, val_loader, criterion, device, features_capture)

        # 记录日志
        log_entry = {
            'epoch': epoch + 1,
            'train_total': avg_loss,
            'train_cls': avg_cls,
            'train_triplet': avg_tri,
            'train_center': avg_center,
            'val_total': val_metrics['total'],
            'val_cls': val_metrics['cls'],
            'val_triplet': val_metrics['triplet'],
            'val_center': val_metrics['center'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        train_log.append(log_entry)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            # 保存 LoRA 权重
            save_lora_weights(model, str(output_dir / "lora_best.pt"))
            # 保存合并后的完整模型
            import copy
            merged_model = copy.deepcopy(model)
            save_merged_model(merged_model, str(output_dir / "best.pth"))
            print(f"  [最佳] epoch {epoch+1}, loss={avg_loss:.4f}")

        # 定期打印
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {avg_loss:.4f} (cls={avg_cls:.4f} tri={avg_tri:.4f} center={avg_center:.4f}) | "
                  f"Val: {val_metrics['total']:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ----------------------------------------------------------
    # 5. 保存最终模型
    # ----------------------------------------------------------
    print("\n保存最终模型...")

    # 保存 LoRA 权重
    save_lora_weights(model, str(output_dir / "lora_last.pt"))

    # 保存合并后的完整模型
    import copy
    final_model = copy.deepcopy(model)
    save_merged_model(final_model, str(output_dir / "last.pth"))

    # 保存训练日志
    log_path = output_dir / "train_log.json"
    with open(str(log_path), 'w') as f:
        json.dump(train_log, f, indent=2)
    print(f"  - 训练日志: {log_path}")

    print(f"\n{'='*60}")
    print("OSNet LoRA 微调训练完成!")
    print(f"{'='*60}")
    print(f"模型保存在: {args.output}/")
    print(f"  - best.pth (合并 LoRA 后的完整模型，可直接用于 ReIDExtractor)")
    print(f"  - last.pth (最终完整模型)")
    print(f"  - lora_best.pt (LoRA 权重备份)")
    print(f"  - lora_last.pt (LoRA 权重备份)")
    print(f"  - train_log.json (训练日志)")
    print(f"\n使用方法:")
    print(f"  在 run_tracking_evaluation.py 中使用 --model custom 即可自动加载微调后的 ReID 权重")


def _register_feature_hook(model: nn.Module, capture: dict):
    """
    在 classifier 层前注册 forward hook，捕获 512 维特征

    这样一次前向传播就能同时获取 logits 和 features，
    避免两次前向传播导致训练时间翻倍。
    """
    def hook_fn(module, input, output):
        # input[0] 就是 classifier 的输入，即 512 维特征
        if isinstance(input, tuple) and len(input) > 0:
            feat = input[0]
            # 如果是 4D (B, C, 1, 1)，需要 squeeze
            if feat.dim() == 4:
                feat = feat.squeeze(-1).squeeze(-1)
            capture['features'] = feat

    # 找到 classifier 层
    for name, module in model.named_modules():
        if name == 'classifier':
            return module.register_forward_hook(hook_fn)

    # 如果没找到 classifier，在最后一个 Linear 层前 hook
    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        return last_linear.register_forward_hook(hook_fn)

    # fallback: 空 hook
    return model.register_forward_hook(lambda m, i, o: None)


if __name__ == "__main__":
    main()
