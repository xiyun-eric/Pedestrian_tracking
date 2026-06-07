"""
自定义损失函数设计

改进方向:
  1. MultiScaleIoULoss: 多尺度 IoU 损失，融合 GIoU + CIoU + EIoU
  2. FocalLoss: 焦点损失，解决正负样本不平衡
  3. MotionConsistencyLoss: 运动一致性损失，约束轨迹平滑性
  4. DepthAwareLoss: 深度感知损失，利用深度信息辅助定位
  5. AdaptiveWeightLoss: 自适应多任务损失权重

设计理念:
  - CIoU 处理框回归的全面性（重叠+中心+宽高比）
  - Focal Loss 处理难易样本不平衡
  - 运动一致性约束时序平滑性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class MultiScaleIoULoss(nn.Module):
    """
    多尺度 IoU 损失
    
    融合三种 IoU 变体:
    - GIoU (Generalized IoU): 引入最小外接矩形，处理无重叠情况
    - CIoU (Complete IoU): 考虑中心距离和宽高比
    - EIoU (Efficient IoU): 分离宽高损失，更高效
    
    最终损失: L = w1*L_GIoU + w2*L_CIoU + w3*L_EIoU
    """
    
    def __init__(
        self,
        w_giou: float = 0.2,
        w_ciou: float = 0.5,
        w_eiou: float = 0.3,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.w_giou = w_giou
        self.w_ciou = w_ciou
        self.w_eiou = w_eiou
        self.reduction = reduction
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) 预测框 [x1, y1, x2, y2]
            target_boxes: (N, 4) 目标框 [x1, y1, x2, y2]
        
        Returns:
            loss: 标量
        """
        # 计算各 IoU 变体
        loss_giou = self._giou_loss(pred_boxes, target_boxes)
        loss_ciou = self._ciou_loss(pred_boxes, target_boxes)
        loss_eiou = self._eiou_loss(pred_boxes, target_boxes)
        
        total_loss = (
            self.w_giou * loss_giou +
            self.w_ciou * loss_ciou +
            self.w_eiou * loss_eiou
        )
        
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        return total_loss
    
    @staticmethod
    def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """计算两个框的 IoU"""
        # 交集
        inter_x1 = torch.max(box1[..., 0], box2[..., 0])
        inter_y1 = torch.max(box1[..., 1], box2[..., 1])
        inter_x2 = torch.min(box1[..., 2], box2[..., 2])
        inter_y2 = torch.min(box1[..., 3], box2[..., 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        # 各自面积
        area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        
        union = area1 + area2 - inter_area
        iou = inter_area / (union + 1e-7)
        
        return iou
    
    def _giou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        GIoU Loss
        
        GIoU = IoU - |C \ (A ∪ B)| / |C|
        其中 C 是包含 A 和 B 的最小外接矩形
        """
        iou = self._box_iou(pred, target)
        
        # 最小外接矩形
        enclose_x1 = torch.min(pred[..., 0], target[..., 0])
        enclose_y1 = torch.min(pred[..., 1], target[..., 1])
        enclose_x2 = torch.max(pred[..., 2], target[..., 2])
        enclose_y2 = torch.max(pred[..., 3], target[..., 3])
        enclose_area = (enclose_x2 - enclose_x1).clamp(0) * (enclose_y2 - enclose_y1).clamp(0)
        
        area1 = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        area2 = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])
        union = area1 + area2 - iou * (area1 + area2 - iou * (area1 + area2))
        
        giou = iou - (enclose_area - union) / (enclose_area + 1e-7)
        return 1 - giou
    
    def _ciou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        CIoU Loss
        
        CIoU = IoU - (ρ²(b, b_gt) / c²) - αv
        其中 ρ² 是中心点距离，c 是对角线长度，v 是宽高比一致性
        """
        iou = self._box_iou(pred, target)
        
        # 中心点
        pred_cx = (pred[..., 0] + pred[..., 2]) / 2
        pred_cy = (pred[..., 1] + pred[..., 3]) / 2
        target_cx = (target[..., 0] + target[..., 2]) / 2
        target_cy = (target[..., 1] + target[..., 3]) / 2
        
        # 中心距离
        rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # 外接矩形对角线
        c_x1 = torch.min(pred[..., 0], target[..., 0])
        c_y1 = torch.min(pred[..., 1], target[..., 1])
        c_x2 = torch.max(pred[..., 2], target[..., 2])
        c_y2 = torch.max(pred[..., 3], target[..., 3])
        c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2
        
        # 宽高比一致性
        pred_w = pred[..., 2] - pred[..., 0]
        pred_h = pred[..., 3] - pred[..., 1]
        target_w = target[..., 2] - target[..., 0]
        target_h = target[..., 3] - target[..., 1]
        
        v = (4 / (math.pi ** 2)) * (
            torch.atan(target_w / (target_h + 1e-7)) -
            torch.atan(pred_w / (pred_h + 1e-7))
        ) ** 2
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)
        
        ciou = iou - rho2 / (c2 + 1e-7) - alpha * v
        return 1 - ciou
    
    def _eiou_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        EIoU Loss
        
        EIoU = IoU - (ρ²(b, b_gt) / c²) - (ρ²(w, w_gt) / Cw²) - (ρ²(h, h_gt) / Ch²)
        直接惩罚宽高差异，比 CIoU 更高效
        """
        iou = self._box_iou(pred, target)
        
        # 中心距离
        pred_cx = (pred[..., 0] + pred[..., 2]) / 2
        pred_cy = (pred[..., 1] + pred[..., 3]) / 2
        target_cx = (target[..., 0] + target[..., 2]) / 2
        target_cy = (target[..., 1] + target[..., 3]) / 2
        
        rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # 外接矩形
        cw = torch.max(pred[..., 2], target[..., 2]) - torch.min(pred[..., 0], target[..., 0])
        ch = torch.max(pred[..., 3], target[..., 3]) - torch.min(pred[..., 1], target[..., 1])
        
        # 宽高差异
        pred_w = pred[..., 2] - pred[..., 0]
        pred_h = pred[..., 3] - pred[..., 1]
        target_w = target[..., 2] - target[..., 0]
        target_h = target[..., 3] - target[..., 1]
        
        rho_w2 = (pred_w - target_w) ** 2
        rho_h2 = (pred_h - target_h) ** 2
        
        eiou = iou - rho2 / (cw ** 2 + ch ** 2 + 1e-7) - \
               rho_w2 / (cw ** 2 + 1e-7) - rho_h2 / (ch ** 2 + 1e-7)
        
        return 1 - eiou


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡
    
    公式: FL = -α_t * (1 - p_t)^γ * log(p_t)
    
    其中:
    - α_t: 类别权重
    - γ: 聚焦参数（越大越关注难样本）
    - p_t: 预测概率
    
    与标准 CrossEntropy 的区别:
    - 对简单样本（p_t高）自动降低权重
    - 对困难样本（p_t低）保持高权重
    - 有效缓解正负样本不平衡
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) 预测 logits
            targets: (N,) 类别标签
        
        Returns:
            loss: 标量
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MotionConsistencyLoss(nn.Module):
    """
    运动一致性损失
    
    约束同一目标在连续帧之间的运动应该是平滑的:
    - 速度变化不应太剧烈
    - 加速度应该有限
    
    L_motion = ||v_t - v_{t-1}||² + λ·||a_t||²
    """
    
    def __init__(self, lambda_acc: float = 0.1):
        super().__init__()
        self.lambda_acc = lambda_acc
        self._prev_velocity = None
    
    def forward(
        self,
        positions: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            positions: (N, T, 2) 连续T帧的位置序列 [cx, cy]
            dt: 时间间隔
        
        Returns:
            loss: 标量
        """
        if positions.shape[1] < 2:
            return torch.tensor(0.0, device=positions.device)
        
        # 计算速度
        velocities = (positions[:, 1:] - positions[:, :-1]) / dt
        
        # 速度平滑性: 相邻速度差
        if velocities.shape[1] >= 2:
            acc = velocities[:, 1:] - velocities[:, :-1]
            acc_loss = (acc ** 2).mean()
        else:
            acc_loss = torch.tensor(0.0, device=positions.device)
        
        # 总损失: 加速度约束
        loss = self.lambda_acc * acc_loss
        
        return loss
    
    def reset(self):
        self._prev_velocity = None


class DepthAwareLoss(nn.Module):
    """
    深度感知边界框损失
    
    原理:
    - 远处的目标框应该更小，置信度应该更低
    - 利用深度信息调整损失权重
    - depth_weight = exp(-depth / max_depth)
    """
    
    def __init__(self, max_depth: float = 80.0):
        super().__init__()
        self.max_depth = max_depth
        self.iou_loss = MultiScaleIoULoss()
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        depths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_boxes: (N, 4) 预测框
            target_boxes: (N, 4) 目标框
            depths: (N,) 深度值（米）
        
        Returns:
            loss: 标量
        """
        iou_loss = self.iou_loss(pred_boxes, target_boxes)
        
        if depths is not None:
            # 深度感知权重: 远处目标权重低
            depth_weight = torch.exp(-depths / self.max_depth)
            iou_loss = (iou_loss * depth_weight).mean()
        
        return iou_loss


class AdaptiveWeightLoss(nn.Module):
    """
    自适应多任务损失权重
    
    基于同方差不确定性 (Homoscedastic Uncertainty):
    L_total = Σ (1/(2σ_i²)) * L_i + log(σ_i)
    
    自动学习每个任务的权重，无需手动调参。
    """
    
    def __init__(self, num_tasks: int = 3):
        """
        Args:
            num_tasks: 任务数量（如 cls, box, dfl）
        """
        super().__init__()
        # 可学习的对数方差参数
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            losses: 各任务的损失列表
        
        Returns:
            total_loss: 自适应加权的总损失
        """
        total_loss = torch.tensor(0.0, device=losses[0].device)
        
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        
        return total_loss * 0.5
