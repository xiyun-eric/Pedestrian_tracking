"""
多目标跟踪评估模块

评估指标:
  - MOTA (Multiple Object Tracking Accuracy): 综合准确率
    MOTA = 1 - (FP + FN + IDSW) / GT
    反映检测和跟踪的整体性能

  - IDF1 (ID F1 Score): 身份保持 F1 分数
    基于IDTP/IDFP/IDFN的标准计算

  - MOTP (Multiple Object Tracking Precision): 定位精度
    平均边界框重叠率

  - IDSW (Identity Switch): 身份切换次数
  - MT (Mostly Tracked): 大部分跟踪成功 (>80%)
  - ML (Mostly Lost): 大部分跟踪失败 (<20%)
  - Frag (Fragmentation): 轨迹碎片数
  - FP (False Positive): 误检
  - FN (False Negative): 漏检
"""

import numpy as np
import json
import csv
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


@dataclass
class TrackingMetrics:
    """跟踪评估指标"""
    MOTA: float = 0.0       # 多目标跟踪准确率
    MOTP: float = 0.0       # 多目标跟踪精度
    IDF1: float = 0.0       # 身份F1分数
    IDSW: int = 0           # 身份切换次数
    FP: int = 0             # 误检数
    FN: int = 0             # 漏检数
    TP: int = 0             # 正确检测数
    GT: int = 0             # 总GT目标数
    MT: int = 0             # 大部分跟踪成功
    ML: int = 0             # 大部分跟踪失败
    Frag: int = 0           # 轨迹碎片数
    num_frames: int = 0     # 评估帧数
    num_gt_ids: int = 0     # GT目标ID数
    num_pred_ids: int = 0   # 预测轨迹ID数

    @property
    def precision(self) -> float:
        """精确率"""
        return self.TP / max(self.TP + self.FP, 1)

    @property
    def recall(self) -> float:
        """召回率"""
        return self.TP / max(self.TP + self.FN, 1)

    def to_dict(self) -> dict:
        return {
            "MOTA": round(self.MOTA, 4),
            "MOTP": round(self.MOTP, 4),
            "IDF1": round(self.IDF1, 4),
            "IDSW": self.IDSW,
            "FP": self.FP,
            "FN": self.FN,
            "TP": self.TP,
            "GT": self.GT,
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "MT": self.MT,
            "ML": self.ML,
            "Frag": self.Frag,
            "num_frames": self.num_frames,
            "num_gt_ids": self.num_gt_ids,
            "num_pred_ids": self.num_pred_ids,
        }

    def print_report(self):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("  多目标跟踪评估报告")
        print("=" * 60)
        print(f"  MOTA:      {self.MOTA:.2%}  (综合准确率)")
        print(f"  MOTP:      {self.MOTP:.2%}  (定位精度)")
        print(f"  IDF1:      {self.IDF1:.2%}  (身份保持)")
        print(f"  IDSW:      {self.IDSW}     (身份切换)")
        print(f"  Precision: {self.precision:.2%}")
        print(f"  Recall:    {self.recall:.2%}")
        print(f"-" * 60)
        print(f"  TP: {self.TP}  FP: {self.FP}  FN: {self.FN}")
        print(f"  MT: {self.MT}  ML: {self.ML}  Frag: {self.Frag}")
        print(f"-" * 60)
        print(f"  帧数: {self.num_frames}  GT目标数: {self.num_gt_ids}  预测轨迹数: {self.num_pred_ids}")
        print("=" * 60)

    def save_json(self, path: str):
        """保存评估报告为JSON"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"评估报告已保存: {path}")

    def save_csv(self, path: str):
        """保存评估报告为CSV"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=d.keys())
            writer.writeheader()
            writer.writerow(d)
        print(f"评估报告已保存: {path}")


class TrackingEvaluator:
    """
    跟踪评估器

    支持:
      1. YOLO标注转MOT格式GT
      2. 标准MOTA/MOTP/IDF1/IDSW/MT/ML/Frag计算
      3. 报告输出（JSON/CSV）
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        predictions: Dict[int, Dict[int, np.ndarray]],
        ground_truth: Dict[int, Dict[int, np.ndarray]],
        eval_mode: str = "prediction_range",
    ) -> TrackingMetrics:
        """
        评估跟踪性能

        Args:
            predictions: {frame_id: {track_id: [x1,y1,x2,y2]}}
            ground_truth: {frame_id: {obj_id: [x1,y1,x2,y2]}}
            eval_mode: 评估模式
                - "prediction_range": 只评估有预测的帧（默认，适用于部分帧评估）
                - "all": 评估所有帧（适用于完整序列评估）

        Returns:
            TrackingMetrics
        """
        metrics = TrackingMetrics()

        # 根据评估模式确定帧范围
        if eval_mode == "prediction_range":
            # 只评估有预测的帧
            all_frames = sorted(predictions.keys())
        else:
            # 评估所有帧（预测+GT的并集）
            all_frames = sorted(set(list(predictions.keys()) + list(ground_truth.keys())))

        metrics.num_frames = len(all_frames)

        # 收集所有GT和预测ID
        all_gt_ids = set()
        all_pred_ids = set()
        for frame_id in all_frames:
            all_gt_ids.update(ground_truth.get(frame_id, {}).keys())
            all_pred_ids.update(predictions.get(frame_id, {}).keys())
        metrics.num_gt_ids = len(all_gt_ids)
        metrics.num_pred_ids = len(all_pred_ids)

        # GT -> pred 的 ID 映射（用于IDSW检测）
        gt_to_pred_map: Dict[int, int] = {}
        id_switches = 0

        # 用于IDF1计算：统计每个(gt_id, pred_id)对的匹配次数
        id_match_counts: Dict[Tuple[int, int], int] = defaultdict(int)

        # 用于MT/ML/Frag计算：统计每个GT ID的出现帧数和匹配帧数
        gt_total_frames: Dict[int, int] = defaultdict(int)
        gt_matched_frames: Dict[int, int] = defaultdict(int)
        # 用于Frag计算：记录每个GT ID的匹配状态变化
        gt_match_history: Dict[int, List[bool]] = defaultdict(list)

        for frame_id in all_frames:
            pred = predictions.get(frame_id, {})
            gt = ground_truth.get(frame_id, {})

            metrics.GT += len(gt)

            if not pred and not gt:
                continue

            if not pred:
                metrics.FN += len(gt)
                for gid in gt:
                    gt_total_frames[gid] += 1
                    gt_matched_frames[gid] += 0
                    gt_match_history[gid].append(False)
                continue

            if not gt:
                metrics.FP += len(pred)
                continue

            # 构建 IoU 矩阵
            pred_ids = list(pred.keys())
            gt_ids = list(gt.keys())

            iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))

            for i, pid in enumerate(pred_ids):
                for j, gid in enumerate(gt_ids):
                    iou_matrix[i, j] = self._iou(pred[pid], gt[gid])

            # 使用匈牙利算法匹配（全局最优）
            cost_matrix = 1 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_pred = set()
            matched_gt = set()

            for pi, gj in zip(row_ind, col_ind):
                if iou_matrix[pi, gj] >= self.iou_threshold:
                    matched_pred.add(pi)
                    matched_gt.add(gj)

                    pid = pred_ids[pi]
                    gid = gt_ids[gj]

                    # 检查身份切换
                    if gid in gt_to_pred_map:
                        if gt_to_pred_map[gid] != pid:
                            id_switches += 1

                    gt_to_pred_map[gid] = pid
                    metrics.TP += 1

                    # 记录ID匹配（用于IDF1）
                    id_match_counts[(gid, pid)] += 1

                    # MOTP
                    metrics.MOTP += iou_matrix[pi, gj]

            metrics.FP += len(pred) - len(matched_pred)
            metrics.FN += len(gt) - len(matched_gt)

            # MT/ML/Frag统计
            matched_gt_ids = {gt_ids[gj] for gj in matched_gt}
            for gid in gt:
                gt_total_frames[gid] += 1
                is_matched = gid in matched_gt_ids
                gt_matched_frames[gid] += int(is_matched)
                gt_match_history[gid].append(is_matched)

        # 计算MOTA
        if metrics.GT > 0:
            metrics.MOTA = 1 - (metrics.FP + metrics.FN + id_switches) / metrics.GT

        # 计算MOTP
        if metrics.TP > 0:
            metrics.MOTP = metrics.MOTP / metrics.TP

        metrics.IDSW = id_switches

        # 计算标准IDF1
        metrics.IDF1 = self._compute_idf1(id_match_counts, all_gt_ids, all_pred_ids,
                                           ground_truth, predictions, all_frames)

        # 计算MT/ML/Frag
        metrics.MT, metrics.ML, metrics.Frag = self._compute_mt_ml_frag(
            gt_total_frames, gt_matched_frames, gt_match_history
        )

        return metrics

    def _compute_idf1(
        self,
        id_match_counts: Dict[Tuple[int, int], int],
        all_gt_ids: set,
        all_pred_ids: set,
        ground_truth: Dict[int, Dict[int, np.ndarray]],
        predictions: Dict[int, Dict[int, np.ndarray]],
        all_frames: List[int],
    ) -> float:
        """
        标准IDF1计算

        IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)

        其中:
          IDTP = sum over (gt_id, pred_id) pairs: min(gt_frames, pred_frames, match_frames)
          IDFP = sum over pred_ids: pred_frames - IDTP_for_this_pred
          IDFN = sum over gt_ids: gt_frames - IDTP_for_this_gt
        """
        # 统计每个GT ID和pred ID的总出现帧数
        gt_id_frames: Dict[int, int] = defaultdict(int)
        pred_id_frames: Dict[int, int] = defaultdict(int)

        for frame_id in all_frames:
            for gid in ground_truth.get(frame_id, {}):
                gt_id_frames[gid] += 1
            for pid in predictions.get(frame_id, {}):
                pred_id_frames[pid] += 1

        # 找到最优ID映射：每个gt_id匹配到匹配次数最多的pred_id
        gt_to_best_pred: Dict[int, int] = {}
        pred_to_best_gt: Dict[int, int] = {}

        # 先按匹配次数降序排列
        sorted_matches = sorted(id_match_counts.items(), key=lambda x: x[1], reverse=True)

        used_gt = set()
        used_pred = set()
        for (gid, pid), count in sorted_matches:
            if gid not in used_gt and pid not in used_pred:
                gt_to_best_pred[gid] = pid
                pred_to_best_gt[pid] = gid
                used_gt.add(gid)
                used_pred.add(pid)

        # 计算IDTP/IDFP/IDFN
        idtp = 0
        for gid, pid in gt_to_best_pred.items():
            idtp += id_match_counts.get((gid, pid), 0)

        # IDFN: GT ID出现但未被其匹配pred ID覆盖的帧数
        idfn = 0
        for gid in all_gt_ids:
            gt_frames = gt_id_frames[gid]
            if gid in gt_to_best_pred:
                matched_frames = id_match_counts.get((gid, gt_to_best_pred[gid]), 0)
                idfn += gt_frames - matched_frames
            else:
                idfn += gt_frames

        # IDFP: pred ID出现但未被其匹配GT ID覆盖的帧数
        idfp = 0
        for pid in all_pred_ids:
            pred_frames = pred_id_frames[pid]
            if pid in pred_to_best_gt:
                matched_frames = id_match_counts.get((pred_to_best_gt[pid], pid), 0)
                idfp += pred_frames - matched_frames
            else:
                idfp += pred_frames

        # IDF1
        if 2 * idtp + idfp + idfn > 0:
            return 2 * idtp / (2 * idtp + idfp + idfn)
        return 0.0

    def _compute_mt_ml_frag(
        self,
        gt_total_frames: Dict[int, int],
        gt_matched_frames: Dict[int, int],
        gt_match_history: Dict[int, int],
    ) -> Tuple[int, int, int]:
        """
        计算MT/ML/Frag

        MT: GT目标被跟踪>80%的帧数
        ML: GT目标被跟踪<20%的帧数
        Frag: GT目标从匹配变为不匹配的次数
        """
        mt = 0
        ml = 0
        frag = 0

        for gid, total in gt_total_frames.items():
            if total == 0:
                continue
            matched = gt_matched_frames[gid]
            ratio = matched / total

            if ratio > 0.8:
                mt += 1
            elif ratio < 0.2:
                ml += 1

            # Frag: 从匹配变为不匹配的次数
            history = gt_match_history[gid]
            for i in range(1, len(history)):
                if history[i - 1] and not history[i]:
                    frag += 1

        return mt, ml, frag

    @staticmethod
    def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个边界框的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def load_yolo_gt(
        self,
        labels_dir: Path,
        image_size: Tuple[int, int],
        scene_name: Optional[str] = None,
        class_ids: Optional[List[int]] = None,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        加载YOLO格式标注作为GT

        将YOLO标注（归一化坐标）转换为像素坐标的MOT格式GT。
        由于YOLO标注没有全局ID，同一目标在不同帧中会被分配不同ID，
        因此此方法主要用于检测精度评估（MOTA中的FP/FN），
        而非ID一致性评估（IDF1/IDSW需要带ID的GT）。

        Args:
            labels_dir: YOLO标注目录（包含.txt文件），会同时搜索 labels_dir 和 labels_dir/train, labels_dir/val
            image_size: (width, height) 图像尺寸
            scene_name: 场景名称过滤（如'scene1'），None则加载全部
            class_ids: 要加载的类别ID列表，None则加载全部

        Returns:
            {frame_id: {obj_id: [x1, y1, x2, y2]}}
        """
        gt = defaultdict(dict)
        img_w, img_h = image_size

        # 搜索标注文件：支持直接目录和train/val子目录
        search_dirs = [labels_dir]
        for sub in ['train', 'val']:
            sub_dir = labels_dir / sub
            if sub_dir.exists():
                search_dirs.append(sub_dir)

        label_files = []
        for d in search_dirs:
            label_files.extend(sorted(d.glob("*.txt")))

        if not label_files:
            print(f"警告: 未找到标注文件 (搜索目录: {search_dirs})")
            return dict(gt)

        # 按文件名排序，确保帧号连续
        label_files = sorted(label_files, key=lambda f: f.stem)

        for label_file in label_files:
            name = label_file.stem

            # 场景过滤
            if scene_name and not name.startswith(scene_name):
                continue

            # 提取帧号：scene2_frame_000100 -> 100
            # 帧号直接对应视频中的第几帧
            frame_id = None
            try:
                parts = name.split('_frame_')
                if len(parts) == 2:
                    frame_id = int(parts[1])
            except ValueError:
                pass

            if frame_id is None:
                continue

            # 读取标注内容
            frame_objs = []
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])
                    if class_ids and cls_id not in class_ids:
                        continue

                    # YOLO格式: class_id center_x center_y width height (归一化)
                    cx = float(parts[1]) * img_w
                    cy = float(parts[2]) * img_h
                    w = float(parts[3]) * img_w
                    h = float(parts[4]) * img_h

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    frame_objs.append(np.array([x1, y1, x2, y2], dtype=np.float32))

            # 只添加有标注的帧
            if frame_objs:
                for obj_id, bbox in enumerate(frame_objs):
                    gt[frame_id][obj_id] = bbox

        # 对GT进行跨帧ID关联
        gt = self._associate_gt_ids(gt)
        return dict(gt)

    def _associate_gt_ids(
        self,
        raw_gt: Dict[int, Dict[int, np.ndarray]],
        iou_threshold: float = 0.3,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        对无全局ID的GT标注进行跨帧关联，分配全局ID

        使用IoU匹配：相邻帧中IoU最大的框属于同一目标。
        这使得YOLO标注（无全局ID）也能用于IDF1/IDSW等跟踪指标评估。

        Args:
            raw_gt: {frame_id: {local_id: bbox}} 原始GT（每帧独立编号）
            iou_threshold: IoU匹配阈值

        Returns:
            {frame_id: {global_id: bbox}} 带全局ID的GT
        """
        if not raw_gt:
            return raw_gt

        sorted_frames = sorted(raw_gt.keys())
        associated_gt = defaultdict(dict)

        next_global_id = 0
        # 上一帧的 {global_id: bbox}，用于与当前帧匹配
        prev_global_objs: Dict[int, np.ndarray] = {}

        for frame_id in sorted_frames:
            frame_objs = raw_gt[frame_id]
            if not frame_objs:
                continue

            local_ids = list(frame_objs.keys())
            bboxes = [frame_objs[lid] for lid in local_ids]

            if not prev_global_objs:
                # 第一帧：直接分配新ID
                for lid in local_ids:
                    associated_gt[frame_id][next_global_id] = frame_objs[lid]
                    prev_global_objs[next_global_id] = frame_objs[lid]
                    next_global_id += 1
            else:
                # 与上一帧IoU匹配
                prev_gids = list(prev_global_objs.keys())
                prev_bboxes = [prev_global_objs[gid] for gid in prev_gids]

                # 构建IoU矩阵: (当前帧local_id, 上一帧global_id)
                iou_matrix = np.zeros((len(local_ids), len(prev_gids)))
                for i in range(len(local_ids)):
                    for j in range(len(prev_gids)):
                        iou_matrix[i, j] = self._iou(bboxes[i], prev_bboxes[j])

                # 匈牙利算法匹配
                cost_matrix = 1 - iou_matrix
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                matched_local = set()
                local_to_global = {}

                for ri, ci in zip(row_ind, col_ind):
                    if iou_matrix[ri, ci] >= iou_threshold:
                        # 匹配成功：继承上一帧的全局ID
                        local_to_global[local_ids[ri]] = prev_gids[ci]
                        matched_local.add(ri)

                # 未匹配的局部ID分配新全局ID
                for i, lid in enumerate(local_ids):
                    if i not in matched_local:
                        local_to_global[lid] = next_global_id
                        next_global_id += 1

                # 写入关联后的GT
                for lid in local_ids:
                    gid = local_to_global[lid]
                    associated_gt[frame_id][gid] = frame_objs[lid]

                # 更新上一帧信息
                prev_global_objs = {gid: frame_objs[lid] for lid, gid in local_to_global.items()}

        return dict(associated_gt)

    def load_mot_gt(
        self,
        gt_path: Path,
        class_ids: Optional[List[int]] = None,
        min_visibility: float = 0.0,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        加载 MOT Challenge 格式 GT 标注

        MOT 格式: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>,
                   <confidence>, <class>, <visibility>
        坐标为左上角 + 宽高，转换为 [x1, y1, x2, y2]

        Args:
            gt_path: gt.txt 文件路径
            class_ids: 要加载的类别ID列表（MOT17中1=pedestrian, 2=person on vehicle等），None则加载全部
            min_visibility: 最小可见度阈值（0~1），低于此值的GT忽略

        Returns:
            {frame_id: {obj_id: [x1, y1, x2, y2]}}
        """
        gt = defaultdict(dict)

        if not gt_path.exists():
            print(f"警告: MOT GT文件不存在: {gt_path}")
            return dict(gt)

        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 7:
                    continue

                frame_id = int(parts[0])
                obj_id = int(parts[1])
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                confidence = float(parts[6])

                # MOT17 GT中 confidence=0 表示忽略区域
                if confidence == 0:
                    continue

                # 类别过滤
                if len(parts) >= 8 and class_ids is not None:
                    cls_id = int(parts[7])
                    if cls_id not in class_ids:
                        continue

                # 可见度过滤
                if len(parts) >= 9:
                    visibility = float(parts[8])
                    if visibility < min_visibility:
                        continue

                # 转换 (x, y, w, h) -> [x1, y1, x2, y2]
                x1 = bb_left
                y1 = bb_top
                x2 = bb_left + bb_width
                y2 = bb_top + bb_height

                gt[frame_id][obj_id] = np.array([x1, y1, x2, y2], dtype=np.float32)

        print(f"MOT GT加载: {gt_path.name} | {len(gt)}帧, "
              f"共{sum(len(v) for v in gt.values())}个标注, "
              f"{len(set(oid for frame in gt.values() for oid in frame.keys()))}个目标ID")
        return dict(gt)

    def load_kitti_gt(self, label_path: Path) -> Dict[int, Dict[int, np.ndarray]]:
        """
        加载 KITTI Tracking Ground Truth 标签

        Args:
            label_path: KITTI 标签文件路径

        Returns:
            {frame_id: {obj_id: [x1, y1, x2, y2]}}
        """
        gt = defaultdict(dict)

        if not label_path.exists():
            print(f"警告: GT标签文件不存在: {label_path}")
            return dict(gt)

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue

                frame_id = int(parts[0])
                obj_id = int(parts[1])
                obj_type = parts[2]

                if obj_type not in ["Pedestrian", "Person_sitting", "Cyclist"]:
                    continue

                x1 = float(parts[6])
                y1 = float(parts[7])
                x2 = float(parts[8])
                y2 = float(parts[9])

                gt[frame_id][obj_id] = np.array([x1, y1, x2, y2], dtype=np.float32)

        return dict(gt)

    def load_predictions_from_tracks(
        self,
        tracks_per_frame: List[List],
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        从 Track 对象列表加载预测

        Args:
            tracks_per_frame: 每帧的 Track 对象列表

        Returns:
            {frame_id: {track_id: bbox}}
        """
        predictions = {}

        for frame_id, tracks in enumerate(tracks_per_frame):
            frame_preds = {}
            for track in tracks:
                if hasattr(track, 'is_confirmed') and not track.is_confirmed:
                    continue
                bbox = track.get_bbox() if hasattr(track, 'get_bbox') else None
                if bbox is not None:
                    frame_preds[track.track_id] = bbox
            predictions[frame_id] = frame_preds

        return predictions

    def save_predictions_mot(
        self,
        predictions: Dict[int, Dict[int, np.ndarray]],
        output_path: str,
        confidences: Optional[Dict[int, Dict[int, float]]] = None,
    ):
        """
        将预测结果保存为MOT Challenge格式

        格式: <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, <class>, <visibility>

        Args:
            predictions: {frame_id: {track_id: [x1,y1,x2,y2]}}
            output_path: 输出文件路径
            confidences: {frame_id: {track_id: confidence}}（可选）
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for frame_id in sorted(predictions.keys()):
                for track_id, bbox in predictions[frame_id].items():
                    x1, y1, x2, y2 = bbox
                    w = x2 - x1
                    h = y2 - y1
                    conf = 1.0
                    if confidences and frame_id in confidences and track_id in confidences[frame_id]:
                        conf = confidences[frame_id][track_id]
                    # MOT格式: frame从1开始
                    f.write(f"{frame_id+1},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},1,1\n")

        print(f"预测结果已保存(MOT格式): {output_path}")

    def save_gt_mot(
        self,
        ground_truth: Dict[int, Dict[int, np.ndarray]],
        output_path: str,
    ):
        """
        将GT保存为MOT Challenge格式

        Args:
            ground_truth: {frame_id: {obj_id: [x1,y1,x2,y2]}}
            output_path: 输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for frame_id in sorted(ground_truth.keys()):
                for obj_id, bbox in ground_truth[frame_id].items():
                    x1, y1, x2, y2 = bbox
                    w = x2 - x1
                    h = y2 - y1
                    f.write(f"{frame_id+1},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1\n")

        print(f"GT已保存(MOT格式): {output_path}")


def evaluate_tracking(
    video_path: str,
    labels_dir: Optional[str] = None,
    scene_name: Optional[str] = None,
    model_path: str = 'yolo11m.pt',
    preset: str = 'standard',
    use_reid: bool = True,
    reid_model: str = 'osnet_x1_0',
    device: str = 'cpu',
    max_frames: int = 0,
    output_dir: str = 'outputs/evaluation',
    iou_threshold: float = 0.5,
) -> TrackingMetrics:
    """
    端到端跟踪评估：检测 + 跟踪 + 评估

    Args:
        video_path: 视频文件路径
        labels_dir: YOLO格式GT标注目录（可选，无则只输出跟踪结果不评估）
        scene_name: 场景名称过滤（如'scene1'）
        model_path: YOLO模型路径
        preset: 跟踪器预设
        use_reid: 是否使用ReID
        reid_model: ReID模型名称
        device: 计算设备
        max_frames: 最大处理帧数（0=全部）
        output_dir: 输出目录
        iou_threshold: 评估IoU阈值

    Returns:
        TrackingMetrics
    """
    import sys
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import cv2
    from deep_method.tracking import create_tracker
    from deep_method.detector import YOLODetector

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return TrackingMetrics()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames <= 0:
        max_frames = total_frames

    print(f"视频: {video_path.name} | {width}x{height} | {fps}FPS | {total_frames}帧")

    # 加载GT（如果提供）
    gt = {}
    if labels_dir:
        evaluator = TrackingEvaluator(iou_threshold=iou_threshold)
        gt = evaluator.load_yolo_gt(
            Path(labels_dir),
            image_size=(width, height),
            scene_name=scene_name,
            class_ids=[0],  # 行人
        )
        print(f"GT加载: {len(gt)}帧, 共{sum(len(v) for v in gt.values())}个标注")
    else:
        evaluator = TrackingEvaluator(iou_threshold=iou_threshold)

    # 创建检测器和跟踪器
    detector = YOLODetector(
        model_path=model_path,
        device=device,
        conf_threshold=0.1,
        iou_threshold=0.5,
        classes=[0],
    )

    tracker = create_tracker(
        tracker_type='advanced',
        preset=preset,
        use_reid=use_reid,
        reid_model=reid_model,
        device=device,
        use_torchreid=True,
    )

    reid = tracker.reid_extractor if hasattr(tracker, 'reid_extractor') else None

    # 收集跟踪结果
    predictions: Dict[int, Dict[int, np.ndarray]] = {}
    confidences_dict: Dict[int, Dict[int, float]] = {}

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测
        detections, confs, _ = detector.detect(frame)

        # ByteTrack分离
        high_conf_threshold = tracker.config.high_conf_threshold
        if len(detections) > 0:
            high_mask = confs >= high_conf_threshold
            high_dets = detections[high_mask]
            high_confs = confs[high_mask]
            low_dets = detections[~high_mask]
            low_confs = confs[~high_mask]
        else:
            high_dets = detections
            high_confs = confs
            low_dets = np.zeros((0, 4), dtype=np.float32)
            low_confs = np.zeros((0,), dtype=np.float32)

        # ReID
        features = None
        if reid is not None and len(high_dets) > 0:
            features = reid.extract_features_batch(frame, high_dets)

        # 跟踪
        tracks = tracker.update(
            high_dets, high_confs,
            features=features, image=frame,
            low_conf_detections=low_dets if len(low_dets) > 0 else None,
            low_conf_confidences=low_confs if len(low_confs) > 0 else None,
        )

        # 收集结果
        frame_preds = {}
        frame_confs = {}
        for track in tracks:
            if track.is_confirmed and track.time_since_update == 0:
                frame_preds[track.track_id] = track.get_bbox()
                frame_confs[track.track_id] = track.confidence
        predictions[frame_count] = frame_preds
        confidences_dict[frame_count] = frame_confs

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  帧 {frame_count}/{max_frames}")

    cap.release()

    # 保存MOT格式结果
    evaluator.save_predictions_mot(predictions, str(output_dir / "predictions.txt"), confidences_dict)

    # 评估
    metrics = TrackingMetrics()
    if gt:
        metrics = evaluator.evaluate(predictions, gt)
        metrics.print_report()
        metrics.save_json(str(output_dir / "evaluation_report.json"))
        metrics.save_csv(str(output_dir / "evaluation_report.csv"))
        evaluator.save_gt_mot(gt, str(output_dir / "gt.txt"))
    else:
        print("未提供GT标注，仅输出跟踪结果（不评估指标）")
        print(f"  总帧数: {frame_count}")
        print(f"  预测轨迹数: {len(set(pid for p in predictions.values() for pid in p.keys()))}")

    return metrics


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="多目标跟踪评估")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("--labels", type=str, default=None, help="YOLO格式GT标注目录")
    parser.add_argument("--scene", type=str, default=None, help="场景名称过滤（如scene1）")
    parser.add_argument("--model", type=str, default='yolo11m.pt', help="YOLO模型路径")
    parser.add_argument("--preset", type=str, default='standard', help="跟踪器预设")
    parser.add_argument("--reid-model", type=str, default='osnet_x1_0', help="ReID模型")
    parser.add_argument("--no-reid", action='store_true', help="禁用ReID")
    parser.add_argument("--device", type=str, default='cpu', help="计算设备")
    parser.add_argument("--frames", type=int, default=0, help="最大处理帧数（0=全部）")
    parser.add_argument("--output", type=str, default='outputs/evaluation', help="输出目录")
    parser.add_argument("--iou", type=float, default=0.5, help="评估IoU阈值")

    args = parser.parse_args()

    evaluate_tracking(
        video_path=args.video,
        labels_dir=args.labels,
        scene_name=args.scene,
        model_path=args.model,
        preset=args.preset,
        use_reid=not args.no_reid,
        reid_model=args.reid_model,
        device=args.device,
        max_frames=args.frames,
        output_dir=args.output,
        iou_threshold=args.iou,
    )


if __name__ == "__main__":
    main()
