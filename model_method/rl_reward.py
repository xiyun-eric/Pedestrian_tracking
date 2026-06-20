"""
RL 奖励函数模块

用于 GRPO 训练的跟踪任务奖励函数:
- 格式奖励: 输出是否包含有效 bbox 格式
- IoU 奖励: 预测 bbox 与 GT 的重叠度
- 完整性奖励: 检测数量与 GT 的匹配度
- ID 一致性奖励: 跨帧 ID 关联准确性
"""

import re


def parse_bboxes_from_text(text):
    """从文本中解析 bbox, 返回 [(x1,y1,x2,y2), ...]"""
    pattern = r'\((\d+),(\d+)\),\((\d+),(\d+)\)'
    matches = re.findall(pattern, text)
    return [(int(m[0]), int(m[1]), int(m[2]), int(m[3])) for m in matches]


def parse_tracks_from_text(text):
    """从文本中解析跟踪结果, 返回 {track_id: [bbox, ...], ...}"""
    tracks = {}
    lines = text.strip().split('\n')
    for line in lines:
        id_match = re.match(r'ID(\d+)', line.strip())
        if not id_match:
            continue
        track_id = int(id_match.group(1))
        bboxes = parse_bboxes_from_text(line)
        if bboxes:
            if track_id not in tracks:
                tracks[track_id] = []
            tracks[track_id].extend(bboxes)
    return tracks


def compute_iou(pred, gt):
    """计算两个 bbox 的 IoU (0-1000 坐标系)"""
    x1 = max(pred[0], gt[0])
    y1 = max(pred[1], gt[1])
    x2 = min(pred[2], gt[2])
    y2 = min(pred[3], gt[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, (pred[2] - pred[0]) * (pred[3] - pred[1]))
    area2 = max(0, (gt[2] - gt[0]) * (gt[3] - gt[1]))
    union = area1 + area2 - inter
    return inter / max(union, 1e-8)


def compute_id_consistency(pred_tracks, gt_ids):
    """
    计算 ID 一致性

    简化实现: 预测的 track ID 数量与 GT 越接近, 一致性越高
    """
    if not pred_tracks:
        return 0.0

    pred_id_count = len(set(pred_tracks.keys()))
    gt_id_count = len(set(gt_ids))

    if gt_id_count == 0:
        return 0.0

    ratio = min(pred_id_count, gt_id_count) / max(pred_id_count, gt_id_count)
    return ratio


def tracking_reward_fn(prompts, completions, gt_bboxes, gt_ids, **kwargs):
    """
    GRPO 奖励函数

    GRPOTrainer 会自动传入:
    - prompts: List - 提示词列表
    - completions: List[str] - 模型生成的文本列表
    - gt_bboxes: 数据集中的 gt_bboxes 列
    - gt_ids: 数据集中的 gt_ids 列

    Returns:
        List[float] - 每个生成的奖励值, 范围 [-1, 5]
    """
    rewards = []
    for text, gt_bbox, gt_id in zip(completions, gt_bboxes, gt_ids):
        reward = 0.0

        # GRPOTrainer 传入的 completion 可能是 dict 或 str
        if isinstance(text, dict):
            text = text.get("content", str(text))
        elif not isinstance(text, str):
            text = str(text)

        # 1. 格式奖励 (-1 或 +1)
        pred_bboxes = parse_bboxes_from_text(text)
        if not pred_bboxes:
            rewards.append(-1.0)
            continue
        reward += 1.0

        # 2. IoU 奖励 (0-2)
        if gt_bbox:
            ious = [max(compute_iou(p, g) for g in gt_bbox) for p in pred_bboxes]
            avg_iou = sum(ious) / len(ious)
            reward += 2.0 * avg_iou

        # 3. 完整性奖励 (0-1)
        if gt_bbox:
            gt_count = len(gt_bbox)
            count_ratio = min(len(pred_bboxes), gt_count) / max(gt_count, 1)
            reward += 1.0 * count_ratio

        # 4. ID 一致性奖励 (0-1)
        if gt_id:
            pred_tracks = parse_tracks_from_text(text)
            consistency = compute_id_consistency(pred_tracks, gt_id)
            reward += 1.0 * consistency

        rewards.append(reward)

    return rewards
