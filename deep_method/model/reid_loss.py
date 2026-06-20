"""
ReID (行人重识别) 专用损失函数

设计理念:
  ReID 任务的核心是学习区分性特征，使得:
  - 同一行人的特征距离近（类内紧凑）
  - 不同行人的特征距离远（类间分离）

损失函数设计:
  1. TripletLoss: 度量学习经典损失，拉近正样本对、推远负样本对
  2. CenterLoss: 学习类中心，增强类内紧凑性
  3. CrossEntropyLabelSmooth: 标签平滑交叉熵，防止过拟合
  4. CombinedReIDLoss: 融合上述损失的联合训练损失

训练策略:
  L_total = w_cls * L_cls + w_tri * L_tri + w_center * L_center
  - L_cls: 分类损失（身份识别）
  - L_tri: 三元组损失（度量学习）
  - L_center: 中心损失（特征紧凑性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TripletLoss(nn.Module):
    """
    Triplet Loss - 度量学习核心损失

    给定锚点 a、正样本 p、负样本 n:
      L = max(0, d(a,p) - d(a,n) + margin)

    其中 d(·,·) 是距离函数（默认欧氏距离），margin 是间隔阈值。

    采矿策略:
      - batch_hard: 在 batch 内为每个锚点选择最远的正样本和最近的负样本
    """

    def __init__(
        self,
        margin: float = 0.3,
        mining_type: str = 'batch_hard',
        distance_type: str = 'euclidean',
    ):
        """
        Args:
            margin: 三元组间隔
            mining_type: 采矿策略 ('batch_hard')
            distance_type: 距离类型 ('euclidean', 'cosine')
        """
        super().__init__()
        self.margin = margin
        self.mining_type = mining_type
        self.distance_type = distance_type

    def _compute_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算特征间的距离矩阵

        Args:
            features: (N, D) 特征矩阵

        Returns:
            dist_mat: (N, N) 距离矩阵
        """
        if self.distance_type == 'cosine':
            # 余弦距离 = 1 - 余弦相似度
            features_norm = F.normalize(features, p=2, dim=1)
            dist_mat = 1 - torch.mm(features_norm, features_norm.t())
        else:
            # 欧氏距离平方
            dist = torch.cdist(features, features, p=2)
            dist_mat = dist ** 2
        return dist_mat

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (N, D) 特征向量（已归一化）
            labels: (N,) 身份标签

        Returns:
            loss: 标量
        """
        if features.shape[0] < 2:
            return torch.tensor(0.0, device=features.device)

        dist_mat = self._compute_distance(features)

        # 构建同身份掩码
        labels = labels.unsqueeze(1)
        is_pos = (labels == labels.t()).float()  # 同身份
        is_neg = (labels != labels.t()).float()  # 不同身份

        # 排除自身
        eye = torch.eye(features.shape[0], device=features.device)
        is_pos = is_pos - eye

        # batch_hard: 最远正样本 + 最近负样本
        # 正样本距离: 同身份中最远的
        dist_ap = (dist_mat * is_pos).max(dim=1)[0]
        # 负样本距离: 不同身份中最近的
        dist_an = (dist_mat + is_pos * 1e6).min(dim=1)[0]

        # Triplet 损失
        loss = F.relu(dist_ap - dist_an + self.margin)

        # 只计算有正样本对的样本
        valid = (is_pos.sum(dim=1) > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        loss = loss[valid].mean()
        return loss


class CenterLoss(nn.Module):
    """
    Center Loss - 学习类中心，增强类内紧凑性

    原理:
      为每个身份维护一个中心向量，训练时同时更新中心和模型参数。
      L_center = 0.5 * Σ ||f_i - c_{y_i}||²

    中心更新规则:
      c_j = c_j - lr_center * Σ (c_j - f_i)  (对属于类 j 的样本)

    优点:
      与 Triplet Loss 互补，Triplet 关注类间分离，Center 关注类内紧凑
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        lr_center: float = 0.5,
    ):
        """
        Args:
            num_classes: 类别数（身份数）
            feat_dim: 特征维度
            lr_center: 中心学习率
        """
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lr_center = lr_center

        # 类中心（不参与梯度计算，手动更新）
        self.centers = nn.Parameter(torch.zeros(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (N, D) 特征向量
            labels: (N,) 身份标签

        Returns:
            loss: 标量
        """
        batch_size = features.shape[0]

        # 确保 centers 和 labels 在同一设备上
        centers = self.centers.to(features.device)

        # 获取对应中心
        centers_batch = centers[labels]

        # 计算损失
        loss = F.mse_loss(features, centers_batch.detach())

        # 手动更新中心（不通过梯度）
        if self.training:
            labels_cpu = labels.cpu()
            features_cpu = features.detach().cpu()
            # 统计每个类别的样本数
            unique_labels = labels_cpu.unique()
            for label in unique_labels:
                mask = (labels_cpu == label)
                if mask.sum() > 0:
                    center_update = (self.centers.data[label] - features_cpu[mask].mean(dim=0))
                    self.centers.data[label] -= self.lr_center * center_update

        return loss


class CrossEntropyLabelSmooth(nn.Module):
    """
    标签平滑交叉熵损失

    原理:
      将硬标签 [0, 0, 1, 0, ...] 软化为:
        y_k = (1 - ε) * δ(k=y) + ε / K

    优点:
      - 防止模型对训练标签过度自信
      - 提高泛化能力
      - 对 ReID 特别有效（行人外观相似，标签噪声大）
    """

    def __init__(
        self,
        num_classes: int,
        epsilon: float = 0.1,
    ):
        """
        Args:
            num_classes: 类别数
            epsilon: 平滑系数
        """
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

        # 预计算平滑标签的对数
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) 分类 logits
            labels: (N,) 类别标签

        Returns:
            loss: 标量
        """
        log_probs = self.log_softmax(inputs)

        # 平滑标签
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.epsilon / (self.num_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.epsilon)

        loss = (-smooth_labels * log_probs).sum(dim=1).mean()
        return loss


class CombinedReIDLoss(nn.Module):
    """
    联合 ReID 训练损失

    L_total = w_cls * L_cls + w_tri * L_tri + w_center * L_center

    其中:
    - L_cls: 标签平滑交叉熵（身份分类）
    - L_tri: Batch-Hard Triplet Loss（度量学习）
    - L_center: Center Loss（特征紧凑性）

    训练阶段策略:
      - 前期（epoch < warmup）: 主要靠分类损失建立基本特征
      - 后期: 三元组损失主导，精修特征空间
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 512,
        w_cls: float = 1.0,
        w_tri: float = 1.0,
        w_center: float = 0.0005,
        triplet_margin: float = 0.3,
        label_smooth_epsilon: float = 0.1,
        center_lr: float = 0.5,
    ):
        """
        Args:
            num_classes: 身份数量
            feat_dim: 特征维度
            w_cls: 分类损失权重
            w_tri: 三元组损失权重
            w_center: 中心损失权重
            triplet_margin: 三元组间隔
            label_smooth_epsilon: 标签平滑系数
            center_lr: 中心学习率
        """
        super().__init__()

        self.w_cls = w_cls
        self.w_tri = w_tri
        self.w_center = w_center

        self.cls_loss = CrossEntropyLabelSmooth(
            num_classes=num_classes,
            epsilon=label_smooth_epsilon,
        )
        self.triplet_loss = TripletLoss(
            margin=triplet_margin,
            mining_type='batch_hard',
        )
        self.center_loss = CenterLoss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            lr_center=center_lr,
        )

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        Args:
            logits: (N, C) 分类器输出
            features: (N, D) 特征向量（归一化前）
            labels: (N,) 身份标签

        Returns:
            dict: {
                'total': 总损失,
                'cls': 分类损失,
                'triplet': 三元组损失,
                'center': 中心损失,
            }
        """
        # 当 w_cls=0 时跳过分类损失（用于 classifier 保持预训练权重的情况）
        if self.w_cls > 0:
            loss_cls = self.cls_loss(logits, labels)
        else:
            loss_cls = torch.tensor(0.0, device=features.device)

        loss_tri = self.triplet_loss(features, labels)
        loss_center = self.center_loss(features, labels)

        total = (
            self.w_cls * loss_cls +
            self.w_tri * loss_tri +
            self.w_center * loss_center
        )

        return {
            'total': total,
            'cls': loss_cls,
            'triplet': loss_tri,
            'center': loss_center,
        }
