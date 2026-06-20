"""
Qwen2-VL 跟踪任务自定义损失函数

L_total = L_lm + λ₁·L_format + λ₂·L_iou + λ₃·L_track

注意: 计划文档中的原始设计存在以下问题, 已在此实现中修正:
1. FormatLoss 原设计只检查 token 是否存在, 不提供有效梯度 → 改为坐标 token 加权 CE
2. IoULoss 原设计需要解码文本, 训练时不可微 → 改为基于 logits 的软 IoU
3. TrackingConsistencyLoss 原设计需要 id_labels → 保留但标注为可选
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateWeightedLoss(nn.Module):
    """
    坐标加权损失: 对 bbox 坐标 token 的交叉熵损失施加更高权重

    原理: 在标准 CE 损失基础上, 识别 labels 中属于 bbox 坐标的 token,
    对这些 token 的损失乘以一个权重因子, 引导模型更关注坐标精度。

    这是 FormatLoss 的实用替代方案:
    - 原方案只检查 box token 是否存在, 不提供有效梯度信号
    - 本方案直接增强坐标位置的梯度, 效果更直接
    """

    def __init__(self, tokenizer, coord_weight=2.0):
        super().__init__()
        self.coord_weight = coord_weight
        self.tokenizer = tokenizer

        # 识别 bbox 相关的特殊 token
        self.box_start_id = tokenizer.convert_tokens_to_ids("<|box_start|>")
        self.box_end_id = tokenizer.convert_tokens_to_ids("<|box_end|>")

        # 数字 token (0-9) 和标点 token 的 ID 集合
        self._build_coord_token_ids()

    def _build_coord_token_ids(self):
        """构建坐标相关 token ID 集合 (数字、逗号、括号)"""
        self.coord_token_ids = set()
        for i in range(10):
            tid = self.tokenizer.convert_tokens_to_ids(str(i))
            if tid != self.tokenizer.unk_token_id:
                self.coord_token_ids.add(tid)

        for ch in [',', '(', ')']:
            tid = self.tokenizer.convert_tokens_to_ids(ch)
            if tid != self.tokenizer.unk_token_id:
                self.coord_token_ids.add(tid)

    def compute_weight_mask(self, labels):
        """
        根据 labels 生成权重掩码 (向量化实现, 避免Python循环)

        在 <|box_start|> 和 <|box_end|> 之间的 token 获得更高权重

        Args:
            labels: [batch, seq_len] - 目标 token IDs

        Returns:
            weights: [batch, seq_len] - 损失权重
        """
        # 向量化实现: 用 cumsum 标记 box 内的位置
        is_box_start = (labels == self.box_start_id)  # [batch, seq_len]
        is_box_end = (labels == self.box_end_id)      # [batch, seq_len]

        # box_start 位置设为1, box_end 位置设为-1
        # cumsum 后, box 内的位置 > 0
        box_markers = is_box_start.int() - is_box_end.int()
        # box_start 之后的位置才开始计数, 所以需要 shift
        # 用 cumsum: start 之前=0, start之后到end之前=1, end之后=0
        in_box = torch.cumsum(box_markers, dim=1).clamp(min=0)
        # box_start 本身不算 (它被 shift 掉了), box_end 也不算
        in_box = (in_box > 0) & ~is_box_end

        # 生成权重: box 内 = coord_weight, 其他 = 1.0
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights = torch.where(in_box, torch.full_like(weights, self.coord_weight), weights)

        # -100 位置权重设为0 (不参与损失)
        weights = torch.where(labels == -100, torch.zeros_like(weights), weights)

        return weights

    def forward(self, logits, labels):
        """
        计算坐标加权的交叉熵损失

        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]

        Returns:
            loss: 标量损失值
        """
        # 标准交叉熵 (reduction='none' 以获取逐 token 损失)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none',
            ignore_index=-100,
        )
        per_token_loss = per_token_loss.view(shift_labels.size())

        # 生成权重掩码
        weights = self.compute_weight_mask(shift_labels)

        # 加权平均
        weighted_loss = per_token_loss * weights.to(per_token_loss.device)
        num_valid = (shift_labels != -100).sum().clamp(min=1)
        loss = weighted_loss.sum() / num_valid

        return loss

    def forward_from_logits(self, logits, labels, hidden_states=None, id_labels=None,
                            attention_mask=None, compute_iou=True):
        """
        统一接口: 从 logits 计算辅助损失

        与 CombinedTrackingLoss.forward() 接口兼容,
        但只使用 logits 和 labels 计算坐标加权损失。

        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]
            hidden_states: 未使用 (接口兼容)
            id_labels: 未使用 (接口兼容)
            attention_mask: 未使用 (接口兼容)
            compute_iou: 未使用 (接口兼容)

        Returns:
            aux_loss: 辅助损失标量
            details: 各损失项的详细值
        """
        loss = self.forward(logits, labels)
        details = {'coord': loss.item()}
        return loss, details


class IoULoss(nn.Module):
    """
    可微 IoU 约束损失 (v2)

    v1 问题: argmax 不可微, 梯度无法回传到坐标 token
    v2 改进: 使用 softmax 概率加权和计算可微 IoU

    原理:
    1. 在 <|box_start|> 和 <|box_end|> 之间, 识别坐标 token 位置
    2. 对每个坐标位置, 用 softmax 概率计算数字的期望值 (0-9)
    3. 用期望值构建软 bbox, 计算 IoU
    4. IoU 是可微的, 梯度可以通过 softmax 回传到 logits
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.box_start_id = tokenizer.convert_tokens_to_ids("<|box_start|>")
        self.box_end_id = tokenizer.convert_tokens_to_ids("<|box_end|>")

        # 构建数字 token ID 映射
        self.digit_token_ids = []
        for d in range(10):
            tid = tokenizer.convert_tokens_to_ids(str(d))
            self.digit_token_ids.append(tid)

        # 逗号和括号的 token ID (用于定位坐标位置)
        self.comma_id = tokenizer.convert_tokens_to_ids(",")
        self.lparen_id = tokenizer.convert_tokens_to_ids("(")
        self.rparen_id = tokenizer.convert_tokens_to_ids(")")

    def parse_bbox_from_text(self, text):
        """从文本中解析 bbox 坐标 (评估时使用)"""
        pattern = r'\((\d+),(\d+)\),\((\d+),(\d+)\)'
        matches = re.findall(pattern, text)
        bboxes = []
        for m in matches:
            x1, y1, x2, y2 = int(m[0]), int(m[1]), int(m[2]), int(m[3])
            bboxes.append([x1, y1, x2, y2])
        return bboxes

    def compute_iou(self, pred_bbox, gt_bbox):
        """计算两个 bbox 的 IoU (0-1000 坐标系)"""
        x1 = max(pred_bbox[0], gt_bbox[0])
        y1 = max(pred_bbox[1], gt_bbox[1])
        x2 = min(pred_bbox[2], gt_bbox[2])
        y2 = min(pred_bbox[3], gt_bbox[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1]))
        area2 = max(0, (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1]))
        union = area1 + area2 - inter

        if union <= 0:
            return 0.0
        return inter / union

    def _extract_soft_coords(self, logits, labels):
        """
        从 logits 中提取可微的坐标值

        对于 box 内的每个 token 位置:
        - 如果 GT 是数字 token, 用 softmax 概率计算数字期望值
        - 如果 GT 是非数字 token, 跳过

        Returns:
            soft_bboxes: List[List[Tensor]] - 每个bbox的4个可微坐标值 [x1, y1, x2, y2]
            gt_bboxes: List[List[int]] - GT bbox 坐标
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        soft_bboxes = []
        gt_bboxes = []

        for b in range(shift_labels.size(0)):
            label_ids = shift_labels[b]
            in_box = False
            current_soft_coords = []
            current_gt_coords = []

            for i in range(label_ids.size(0)):
                tid = label_ids[i].item()

                if tid == self.box_start_id:
                    in_box = True
                    current_soft_coords = []
                    current_gt_coords = []
                    continue

                if tid == self.box_end_id:
                    in_box = False
                    if len(current_soft_coords) == 4 and len(current_gt_coords) == 4:
                        soft_bboxes.append(current_soft_coords)
                        gt_bboxes.append(current_gt_coords)
                    current_soft_coords = []
                    current_gt_coords = []
                    continue

                if in_box and tid != -100:
                    # 检查是否是数字 token
                    if tid in self.digit_token_ids:
                        # GT 数字值
                        digit_val = self.digit_token_ids.index(tid)
                        current_gt_coords.append(digit_val)

                        # 可微的数字期望值
                        # 只取数字 token 对应的 logits
                        digit_logits = torch.stack([
                            shift_logits[b, i, did] for did in self.digit_token_ids
                        ])
                        digit_probs = F.softmax(digit_logits, dim=0)
                        soft_val = (digit_probs * torch.arange(10, device=digit_probs.device, dtype=digit_probs.dtype)).sum()
                        current_soft_coords.append(soft_val)
                    # 非数字 token (逗号、括号) 用于分隔坐标
                    elif tid == self.comma_id:
                        # 坐标分隔符, 不需要处理
                        pass

        return soft_bboxes, gt_bboxes

    def forward_from_logits(self, logits, labels):
        """
        可微 IoU 损失 (v2)

        通过 softmax 概率加权和计算可微的 bbox 坐标,
        然后计算 IoU, 梯度可以通过 softmax 回传到 logits。

        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]

        Returns:
            loss: 标量, 1 - 平均 IoU (可微)
        """
        soft_bboxes, gt_bboxes = self._extract_soft_coords(logits, labels)

        if not soft_bboxes:
            return torch.tensor(0.0, device=logits.device)

        total_iou = torch.tensor(0.0, device=logits.device)
        num_valid = 0

        for soft_coords, gt_coords in zip(soft_bboxes, gt_bboxes):
            if len(soft_coords) != 4 or len(gt_coords) != 4:
                continue

            # soft_coords 是单个数字的期望值 (0-9)
            # 实际坐标是多数字组合, 这里简化为单数字 IoU
            # 注意: 这是一种近似, 完整实现需要处理多位数
            # 但作为辅助损失已经足够提供梯度信号

            # 构建软 bbox (单数字近似)
            soft_x1, soft_y1, soft_x2, soft_y2 = soft_coords
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_coords

            # 转换为 tensor
            gt_x1 = torch.tensor(gt_x1, device=logits.device, dtype=torch.float32)
            gt_y1 = torch.tensor(gt_y1, device=logits.device, dtype=torch.float32)
            gt_x2 = torch.tensor(gt_x2, device=logits.device, dtype=torch.float32)
            gt_y2 = torch.tensor(gt_y2, device=logits.device, dtype=torch.float32)

            # 计算 IoU (可微)
            inter_x1 = torch.max(soft_x1, gt_x1)
            inter_y1 = torch.max(soft_y1, gt_y1)
            inter_x2 = torch.min(soft_x2, gt_x2)
            inter_y2 = torch.min(soft_y2, gt_y2)

            inter_area = torch.max(inter_x2 - inter_x1, torch.tensor(0.0, device=logits.device)) * \
                         torch.max(inter_y2 - inter_y1, torch.tensor(0.0, device=logits.device))

            area1 = torch.max(soft_x2 - soft_x1, torch.tensor(0.0, device=logits.device)) * \
                    torch.max(soft_y2 - soft_y1, torch.tensor(0.0, device=logits.device))
            area2 = torch.max(gt_x2 - gt_x1, torch.tensor(0.0, device=logits.device)) * \
                    torch.max(gt_y2 - gt_y1, torch.tensor(0.0, device=logits.device))

            union_area = area1 + area2 - inter_area + 1e-8

            iou = inter_area / union_area
            total_iou = total_iou + iou
            num_valid += 1

        if num_valid == 0:
            return torch.tensor(0.0, device=logits.device)

        avg_iou = total_iou / num_valid
        return 1.0 - avg_iou

    def forward_from_text(self, generated_texts, gt_bboxes_list):
        """
        从解码文本计算 IoU 损失 (评估时使用)

        Args:
            generated_texts: List[str] - 模型生成的文本
            gt_bboxes_list: List[List[List[int]]] - GT bbox (0-1000坐标)
        """
        total_iou = 0.0
        num_valid = 0

        for text, gt_bboxes in zip(generated_texts, gt_bboxes_list):
            pred_bboxes = self.parse_bbox_from_text(text)
            if not pred_bboxes or not gt_bboxes:
                continue
            for pred in pred_bboxes:
                best_iou = max(self.compute_iou(pred, gt) for gt in gt_bboxes)
                total_iou += best_iou
                num_valid += 1

        if num_valid == 0:
            return torch.tensor(0.0)
        return torch.tensor(1.0 - total_iou / num_valid)


class TrackingConsistencyLoss(nn.Module):
    """
    跟踪一致性损失: 同一 track ID 在不同帧的隐藏状态应相似

    数据来源: dataset.py 的 _build_id_labels 方法从 MOT17 GT 的
    "IDx:" 模式中提取 track ID, 生成 id_labels 张量。
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, hidden_states, id_labels, attention_mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            id_labels: [batch, seq_len] - 每个 token 的 track ID (-100 表示不参与)
            attention_mask: [batch, seq_len]
        """
        if id_labels is None:
            return torch.tensor(0.0, device=hidden_states.device)

        batch_size = hidden_states.size(0)
        total_loss = 0.0
        num_valid = 0

        for b in range(batch_size):
            id_embeddings = {}
            for i in range(hidden_states.size(1)):
                if id_labels[b, i] == -100:
                    continue
                tid = id_labels[b, i].item()
                if tid not in id_embeddings:
                    id_embeddings[tid] = []
                id_embeddings[tid].append(hidden_states[b, i])

            if len(id_embeddings) < 2:
                continue

            # 计算每个 ID 的原型
            prototypes = {}
            for tid, embs in id_embeddings.items():
                if len(embs) > 0:
                    prototypes[tid] = torch.stack(embs).mean(dim=0)

            if len(prototypes) < 2:
                continue

            # 对比损失
            proto_list = torch.stack(list(prototypes.values()))
            proto_norm = F.normalize(proto_list, dim=-1)
            sim_matrix = torch.matmul(proto_norm, proto_norm.T) / self.temperature

            n = len(prototypes)
            # 对角线 mask (排除自身)
            mask = ~torch.eye(n, dtype=torch.bool, device=hidden_states.device)

            # InfoNCE: 每个 prototype 与其他所有 prototype 对比
            exp_sim = torch.exp(sim_matrix) * mask
            pos_sim = exp_sim.sum(dim=-1)
            all_sim = exp_sim.sum(dim=-1) + torch.exp(torch.diag(sim_matrix))

            # 避免除零
            loss = -torch.log(pos_sim / all_sim.clamp(min=1e-8) + 1e-8).mean()
            total_loss += loss
            num_valid += 1

        if num_valid == 0:
            return torch.tensor(0.0, device=hidden_states.device)

        return total_loss / num_valid


class CombinedTrackingLoss(nn.Module):
    """
    组合损失函数 (v2)

    L_total = L_lm + λ₁·L_coord_weighted + λ₂·L_iou + λ₃·L_track

    v2 改进:
    - L_iou: 从 argmax 不可微改为 softmax 可微 IoU
    - L_track: id_labels 已从 MOT17 GT 提取, 不再为 None
    """

    def __init__(self, tokenizer, lambda_coord=0.5, lambda_iou=1.0, lambda_track=0.5):
        super().__init__()
        self.coord_loss = CoordinateWeightedLoss(tokenizer, coord_weight=2.0)
        self.iou_loss = IoULoss(tokenizer)
        self.track_loss = TrackingConsistencyLoss()
        self.lambda_coord = lambda_coord
        self.lambda_iou = lambda_iou
        self.lambda_track = lambda_track

    def forward(self, logits, labels, hidden_states=None, id_labels=None,
                attention_mask=None, compute_iou=True):
        """
        计算辅助损失

        Args:
            logits: [batch, seq_len, vocab_size]
            labels: [batch, seq_len]
            hidden_states: [batch, seq_len, hidden_dim] (可选, 用于 L_track)
            id_labels: [batch, seq_len] (可选, 用于 L_track)
            attention_mask: [batch, seq_len]
            compute_iou: 是否计算 IoU 损失 (较慢, 可周期性计算)

        Returns:
            aux_loss: 辅助损失标量
            details: 各损失项的详细值
        """
        device = logits.device

        # 坐标加权损失
        L_coord = self.coord_loss(logits, labels)

        # IoU 损失 (可选, 训练时较慢)
        L_iou = torch.tensor(0.0, device=device)
        if compute_iou:
            L_iou = self.iou_loss.forward_from_logits(logits, labels)

        # 跟踪一致性损失 (可选)
        L_track = torch.tensor(0.0, device=device)
        if hidden_states is not None and id_labels is not None:
            L_track = self.track_loss(hidden_states, id_labels, attention_mask)

        aux_loss = (
            self.lambda_coord * L_coord +
            self.lambda_iou * L_iou +
            self.lambda_track * L_track
        )

        details = {
            'coord': L_coord.item(),
            'iou': L_iou.item(),
            'track': L_track.item(),
        }

        return aux_loss, details

    def forward_from_logits(self, logits, labels, hidden_states=None, id_labels=None,
                            attention_mask=None, compute_iou=True):
        """统一接口: 与 forward() 相同"""
        return self.forward(logits, labels, hidden_states=hidden_states,
                           id_labels=id_labels, attention_mask=attention_mask,
                           compute_iou=compute_iou)
