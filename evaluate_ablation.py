"""消融实验评估脚本 - 使用 motmetrics 计算 MOTA 等指标"""
import motmetrics as mm
import numpy as np
from pathlib import Path
from collections import defaultdict

mm.lap.default_solver = 'scipy'


def load_mot_file(path):
    """加载 MOT 格式文件: frame, id, x, y, w, h, conf, class, vis"""
    data = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            tid = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            data[frame].append((tid, x, y, w, h))
    return data


def evaluate(pred_path, gt_path, max_frame=300):
    """评估预测结果"""
    pred_data = load_mot_file(pred_path)
    gt_data = load_mot_file(gt_path)

    # 只评估前 max_frame 帧
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_data.keys())))
    all_frames = [f for f in all_frames if f <= max_frame]

    # 创建 accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    for frame in all_frames:
        gt_boxes = []
        gt_ids = []
        for tid, x, y, w, h in gt_data.get(frame, []):
            gt_ids.append(tid)
            gt_boxes.append([x, y, w, h])

        pred_boxes = []
        pred_ids = []
        for tid, x, y, w, h in pred_data.get(frame, []):
            pred_ids.append(tid)
            pred_boxes.append([x, y, w, h])

        # 转换为 motmetrics 格式 (x, y, w, h) -> [x, y, w, h]
        if gt_boxes and pred_boxes:
            gt_arr = np.array(gt_boxes)
            pred_arr = np.array(pred_boxes)
            # motmetrics 期望格式: [x, y, w, h]
            distances = mm.distances.iou_matrix(gt_arr, pred_arr, max_iou=0.5)
            acc.update(gt_ids, pred_ids, distances)
        elif gt_boxes:
            gt_arr = np.array(gt_boxes)
            acc.update(gt_ids, [], np.zeros((len(gt_ids), 0)))
        elif pred_boxes:
            pred_arr = np.array(pred_boxes)
            acc.update([], pred_ids, np.zeros((0, len(pred_ids))))

    # 计算指标
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'num_frames', 'num_matches', 'num_false_positives', 'num_misses',
        'mota', 'motp', 'precision', 'recall',
    ], name='result')

    return summary


def main():
    gt_path = 'data/MOT17/MOT17-11-FRCNN/gt/gt.txt'
    max_frame = 300

    configs = [
        ('完整', 'outputs/ablation/full/seq_MOT17-11-FRCNN/predictions.txt'),
        ('无光流', 'outputs/ablation/no_flow/seq_MOT17-11-FRCNN/predictions.txt'),
        ('无ReID', 'outputs/ablation/no_reid/seq_MOT17-11-FRCNN/predictions.txt'),
        ('纯卡尔曼', 'outputs/ablation/pure_kalman/seq_MOT17-11-FRCNN/predictions.txt'),
    ]

    summaries = []
    for name, pred_path in configs:
        if not Path(pred_path).exists():
            print(f"[跳过] {name}: 文件不存在 {pred_path}")
            continue
        print(f"[评估] {name} ...")
        summary = evaluate(pred_path, gt_path, max_frame)
        summary.index = [name]
        summaries.append(summary)

    if summaries:
        # 合并所有结果
        import pandas as pd
        all_summary = pd.concat(summaries)
        # 格式化输出
        print("\n" + "=" * 80)
        print("消融实验评估结果 (MOT17-11, 前300帧)")
        print("=" * 80)
        print(all_summary.to_string())
        print("=" * 80)


if __name__ == '__main__':
    main()
