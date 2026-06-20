"""
传统方法跟踪器（卡尔曼滤波 + 级联匹配 + 匈牙利算法）

从 code/Tracking/tracker.py 迁移，去除深度信息依赖，
作为传统方法的专用跟踪器，不依赖 deep_method 模块。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum

from traditional_method.kalman_filter import KalmanFilter


class TrackState(Enum):
    """轨迹状态枚举"""
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class Track:
    """
    单个目标的轨迹类

    属性:
        track_id: 轨迹ID
        state: 当前状态 [x, y, w, h, vx, vy, vw, vh]
        covariance: 状态协方差矩阵
        hits: 连续检测到的帧数
        age: 轨迹存在的总帧数
        time_since_update: 自上次更新以来的帧数
        history: 历史位置记录
    """

    _count = 0

    def __init__(
        self,
        detection: np.ndarray,
        confidence: float = 1.0,
    ):
        """
        初始化轨迹

        Args:
            detection: [x1, y1, x2, y2] 边界框
            confidence: 检测置信度
        """
        self.track_id = Track._count
        Track._count += 1

        self.kf = KalmanFilter()

        z = KalmanFilter.bbox_to_z(detection)
        self.mean, self.covariance = self.kf.initiate(z)

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confidence = confidence
        self.history: List[np.ndarray] = [detection.copy()]

        self.track_state = TrackState.TENTATIVE

        self._class_id = 0
        
        # 记录初始宽高作为尺寸参考基准
        init_w = detection[2] - detection[0]
        init_h = detection[3] - detection[1]
        self._ref_w = init_w
        self._ref_h = init_h
        self._ref_count = 1

    def predict(self):
        """预测下一帧的状态（使用自适应过程噪声）"""
        # 保存预测前的宽高，用于约束
        prev_w = self.mean[2]
        prev_h = self.mean[3]
        
        # 抑制速度分量漂移：
        # vw/vh 强衰减：行人尺寸帧间变化极小
        self.mean[6] *= 0.1  # vw 衰减
        self.mean[7] *= 0.1  # vh 衰减
        # vx/vy 温和衰减：未匹配时速度不确定性增大，防止预测位置漂移
        # 已匹配时(time_since_update==0)保留速度，未匹配时逐渐衰减
        if self.time_since_update > 0:
            self.mean[4] *= 0.7  # vx 衰减
            self.mean[5] *= 0.7  # vy 衰减
        
        # 使用自适应过程噪声：未匹配帧数越多，噪声越大，搜索范围越广
        adaptive_Q = self.kf.get_adaptive_Q(self.time_since_update)
        self.mean, self.covariance = self.kf.predict_with_Q(
            self.mean, self.covariance, adaptive_Q
        )
        self.age += 1
        self.time_since_update += 1

        # 宽高约束：预测后宽高不能突变（每帧最多变化5%）
        max_change = 1.05
        min_change = 0.95
        self.mean[2] = np.clip(self.mean[2], prev_w * min_change, prev_w * max_change)
        self.mean[3] = np.clip(self.mean[3], prev_h * min_change, prev_h * max_change)
        
        # 基于历史参考尺寸的硬约束：宽高不能超过参考尺寸的2倍，不能低于0.4倍
        ref_w = self._ref_w / self._ref_count
        ref_h = self._ref_h / self._ref_count
        self.mean[2] = np.clip(self.mean[2], ref_w * 0.4, ref_w * 2.0)
        self.mean[3] = np.clip(self.mean[3], ref_h * 0.4, ref_h * 2.0)

        w = self.mean[2]
        h = self.mean[3]
        if w <= 0 or h <= 0:
            self.track_state = TrackState.DELETED

    def update(
        self,
        detection: np.ndarray,
        confidence: float = 1.0,
    ):
        """
        使用检测更新轨迹

        Args:
            detection: [x1, y1, x2, y2] 边界框
            confidence: 检测置信度
        """
        z = KalmanFilter.bbox_to_z(detection)
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, z)
        
        # 抑制宽高速度分量：防止卡尔曼增益给vw/vh分配过大值
        self.mean[6] *= 0.1  # vw 衰减
        self.mean[7] *= 0.1  # vh 衰减

        # 宽高约束：更新后宽高不能突变（每帧最多变化10%）
        if len(self.history) > 0:
            prev_bbox = self.history[-1]
            prev_w = prev_bbox[2] - prev_bbox[0]
            prev_h = prev_bbox[3] - prev_bbox[1]
            
            cur_w = self.mean[2]
            cur_h = self.mean[3]
            
            min_w = prev_w * 0.9
            max_w = prev_w * 1.1
            min_h = prev_h * 0.9
            max_h = prev_h * 1.1
            
            self.mean[2] = np.clip(cur_w, min_w, max_w)
            self.mean[3] = np.clip(cur_h, min_h, max_h)
        
        # 基于历史参考尺寸的硬约束
        ref_w = self._ref_w / self._ref_count
        ref_h = self._ref_h / self._ref_count
        self.mean[2] = np.clip(self.mean[2], ref_w * 0.4, ref_w * 2.0)
        self.mean[3] = np.clip(self.mean[3], ref_h * 0.4, ref_h * 2.0)

        self.hits += 1
        self.time_since_update = 0
        self.confidence = confidence
        self.history.append(detection.copy())
        
        # 更新参考尺寸：用指数移动平均平滑
        det_w = detection[2] - detection[0]
        det_h = detection[3] - detection[1]
        alpha = 0.3  # 新检测权重
        self._ref_w = self._ref_w * (1 - alpha) + det_w * alpha
        self._ref_h = self._ref_h * (1 - alpha) + det_h * alpha
        self._ref_count = 1  # ref_w/ref_h 已经是加权平均，不需要除以count

        if self.track_state == TrackState.TENTATIVE and self.hits >= 2:
            self.track_state = TrackState.CONFIRMED

    def mark_missed(self):
        """标记轨迹为丢失"""
        self.track_state = TrackState.DELETED

    def get_bbox(self) -> np.ndarray:
        """获取当前边界框 [x1, y1, x2, y2]"""
        return KalmanFilter.z_to_bbox(self.mean[:4])

    def get_velocity(self) -> np.ndarray:
        """获取当前速度 [vx, vy, vw, vh]"""
        return self.mean[4:8]

    def get_predicted_bbox(self) -> np.ndarray:
        """获取预测的边界框"""
        return self.get_bbox()

    def mahalanobis_distance(
        self,
        measurement: np.ndarray,
        only_position: bool = False
    ) -> float:
        """
        计算与检测的马氏距离

        Args:
            measurement: [x1, y1, x2, y2] 边界框
            only_position: 是否只计算位置距离

        Returns:
            马氏距离
        """
        z = KalmanFilter.bbox_to_z(measurement)
        return self.kf.mahalanobis_distance(
            self.mean, self.covariance, z, only_position
        )

    @property
    def is_tentative(self) -> bool:
        """轨迹是否为暂定状态"""
        return self.track_state == TrackState.TENTATIVE

    @property
    def is_confirmed(self) -> bool:
        """轨迹是否已确认"""
        return self.track_state == TrackState.CONFIRMED

    @property
    def is_deleted(self) -> bool:
        """轨迹是否应被删除"""
        return self.track_state == TrackState.DELETED

    @classmethod
    def reset_count(cls):
        """重置轨迹ID计数器"""
        cls._count = 0


class Tracker:
    """
    多目标跟踪器（级联匹配 + 匈牙利算法）

    使用卡尔曼滤波 + 级联匹配 + 匈牙利算法实现多目标跟踪
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        min_hits: int = 2,
        max_iou_distance: float = 0.7,
        max_age_unmatched: int = 15,
        lambda_iou: float = 0.5,
        flow_correction_weight: float = 0.3,
        tentative_tolerance: int = 2,
    ):
        """
        初始化跟踪器

        Args:
            iou_threshold: IoU匹配阈值
            max_age: 轨迹最大丢失帧数（超过后删除）
            min_hits: 确认轨迹所需的最小检测次数
            max_iou_distance: 最大IoU距离
            max_age_unmatched: 未匹配轨迹最大存活帧数（级联匹配范围）
            lambda_iou: IoU距离权重（马氏距离权重为1-lambda_iou）
            flow_correction_weight: 光流校正权重（0=不校正，1=完全信任光流）
            tentative_tolerance: TENTATIVE轨迹容忍帧数（未匹配多少帧后才删除）
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_iou_distance = max_iou_distance
        self.max_age_unmatched = max_age_unmatched
        self.lambda_iou = lambda_iou
        self.flow_correction_weight = flow_correction_weight
        self.tentative_tolerance = tentative_tolerance

        self.tracks: List[Track] = []
        self.frame_count = 0

        self._gating_threshold = 9.4877

    def update(
        self,
        detections: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        class_ids: Optional[np.ndarray] = None,
        flow_corrections: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> List[Track]:
        """
        更新跟踪器

        Args:
            detections: (N, 4) 边界框数组 [x1, y1, x2, y2]
            confidences: (N,) 置信度数组
            class_ids: (N,) 类别ID数组
            flow_corrections: 光流校正字典 {track_id: (dx, dy)}

        Returns:
            当前帧确认的轨迹列表
        """
        self.frame_count += 1

        if len(detections) == 0:
            detections = np.zeros((0, 4), dtype=np.float32)

        if confidences is None:
            confidences = np.ones(len(detections))

        if class_ids is None:
            class_ids = np.zeros(len(detections), dtype=np.int32)

        # 记录每个轨迹在predict之前是否刚被匹配（time_since_update==0）
        # 用于光流校正判断：只有上一帧未匹配的轨迹才需要光流校正
        was_matched = {}
        for track in self.tracks:
            was_matched[track.track_id] = (track.time_since_update == 0)

        for track in self.tracks:
            track.predict()

        # 光流校正：仅对上一帧未匹配的轨迹使用
        # 上一帧已匹配的轨迹由检测框直接更新，不需要光流校正
        if flow_corrections and self.flow_correction_weight > 0:
            for track in self.tracks:
                if track.track_id not in flow_corrections or track.is_deleted:
                    continue
                # 只有上一帧未匹配的轨迹才用光流校正
                if was_matched.get(track.track_id, False):
                    continue
                
                dx, dy = flow_corrections[track.track_id]
                
                # 保守的光流校正权重
                adaptive_weight = min(
                    self.flow_correction_weight * 0.5,
                    0.2
                )
                
                # 计算光流位移
                correction_x = dx * adaptive_weight
                correction_y = dy * adaptive_weight
                
                # 限制单次校正幅度
                max_correction = 5.0
                correction_x = np.clip(correction_x, -max_correction, max_correction)
                correction_y = np.clip(correction_y, -max_correction, max_correction)
                
                track.mean[0] += correction_x
                track.mean[1] += correction_y

        matched, unmatched_detections, unmatched_tracks = self._cascade_associate(
            detections, confidences
        )

        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(
                detections[det_idx],
                confidences[det_idx],
            )

        # 渐进式删除：TENTATIVE容忍若干帧未匹配，CONFIRMED超过max_age才删除
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            if track.is_tentative and track.time_since_update > self.tentative_tolerance:
                track.mark_missed()
            elif track.is_confirmed and track.time_since_update > self.max_age:
                track.mark_missed()

        for det_idx in unmatched_detections:
            det = detections[det_idx]
            # 检查该检测是否与已有轨迹高度重叠或位于交汇区域
            # 如果重叠，说明是融合框、重复检测或交汇虚假框，不应创建新轨迹
            is_invalid = False
            det_area = max((det[2] - det[0]) * (det[3] - det[1]), 1.0)
            det_cx = (det[0] + det[2]) / 2
            det_cy = (det[1] + det[3]) / 2
            
            for track in self.tracks:
                if track.is_deleted:
                    continue
                track_bbox = track.get_bbox()
                track_area = max((track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1]), 1.0)
                iou = self._compute_iou(track_bbox, det)
                
                # 条件1：与已有轨迹IoU过高，且面积差异不大（正常重叠）
                # 但如果轨迹框异常大（面积>检测3倍），不应抑制新检测
                if iou > 0.3 and track_area < det_area * 3.0:
                    is_invalid = True
                    break
                
                # 条件2：检测框中心在已有轨迹内部，且面积差异大
                if (track_bbox[0] <= det_cx <= track_bbox[2] and
                    track_bbox[1] <= det_cy <= track_bbox[3]):
                    area_ratio = det_area / track_area
                    # 只有当轨迹大小合理时才抑制
                    if 0.5 <= area_ratio <= 2.0:
                        is_invalid = True
                        break
            
            # 条件3：检测框与多个轨迹都有中等重叠（交汇区域特征）
            if not is_invalid and len(self.tracks) > 1:
                overlap_count = 0
                for track in self.tracks:
                    if track.is_deleted:
                        continue
                    track_bbox = track.get_bbox()
                    iou = self._compute_iou(track_bbox, det)
                    if iou > 0.1:
                        overlap_count += 1
                if overlap_count >= 2:
                    is_invalid = True
            
            if not is_invalid:
                new_track = Track(
                    det,
                    confidences[det_idx],
                )
                self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if not t.is_deleted]

        # 重复轨迹合并：同一目标不应被多条轨迹跟踪
        self._merge_duplicate_tracks()

        # 只返回当前帧有检测匹配的轨迹（time_since_update == 0）
        # 未匹配的轨迹仅在内部保留，用于后续帧的重匹配，不参与可视化
        # 这是防止"幽灵框"的关键：没有检测支撑的轨迹不应被显示
        active_tracks = [t for t in self.tracks if t.is_confirmed and t.time_since_update == 0]

        return active_tracks

    def _merge_duplicate_tracks(self):
        """
        合并重复轨迹：当多个轨迹的预测框高度重叠时，
        保留置信度最高（hits最多）的轨迹，删除其余轨迹。
        
        这解决了 HOG 检测器对同一目标产生多个检测框，
        导致多条轨迹跟踪同一目标的问题。
        """
        if len(self.tracks) < 2:
            return

        # 计算所有轨迹对之间的 IoU
        n = len(self.tracks)
        to_remove = set()

        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue

                bbox_i = self.tracks[i].get_bbox()
                bbox_j = self.tracks[j].get_bbox()
                iou = self._compute_iou(bbox_i, bbox_j)

                # IoU > 0.5 认为是同一目标的重复轨迹
                if iou > 0.5:
                    # 保留 hits 更多的轨迹（更稳定）
                    if self.tracks[i].hits >= self.tracks[j].hits:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break  # i 被删除，不需要再比较

        if to_remove:
            self.tracks = [t for idx, t in enumerate(self.tracks) if idx not in to_remove]

    def _cascade_associate(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        级联匹配

        按照轨迹的time_since_update从小到大进行级联匹配，
        优先匹配最近更新的轨迹。

        Args:
            detections: (N, 4) 边界框数组
            confidences: (N,) 置信度数组

        Returns:
            (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed]

        matched = []
        unmatched_detections = list(range(len(detections)))

        for age in range(1, self.max_age_unmatched + 1):
            if len(unmatched_detections) == 0:
                break

            track_indices = [
                i for i in confirmed_tracks
                if self.tracks[i].time_since_update == age
            ]

            if len(track_indices) == 0:
                continue

            track_matched, unmatched_detections, _ = self._min_cost_matching(
                detections, confidences, track_indices, unmatched_detections
            )

            matched.extend(track_matched)

        if len(unconfirmed_tracks) > 0 and len(unmatched_detections) > 0:
            iou_matched, unmatched_detections, _ = self._iou_matching(
                detections, unconfirmed_tracks, unmatched_detections
            )
            matched.extend(iou_matched)

        unmatched_tracks = [
            i for i in range(len(self.tracks))
            if i not in [m[0] for m in matched]
        ]

        return matched, unmatched_detections, unmatched_tracks

    def _min_cost_matching(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
        track_indices: List[int],
        detection_indices: List[int],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        最小代价匹配（马氏距离 + IoU）
        """
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return [], detection_indices, track_indices

        cost_matrix = np.zeros((len(track_indices), len(detection_indices)), dtype=np.float32)

        for i, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            track_bbox = track.get_bbox()
            velocity = track.get_velocity()

            for j, det_idx in enumerate(detection_indices):
                det = detections[det_idx]

                mahal_dist = track.mahalanobis_distance(det, only_position=False)

                iou = self._compute_iou(track_bbox, det)
                iou_dist = 1 - iou

                if mahal_dist > self._gating_threshold:
                    cost_matrix[i, j] = 1e5
                else:
                    base_cost = (
                        self.lambda_iou * iou_dist +
                        (1 - self.lambda_iou) * (mahal_dist / self._gating_threshold)
                    )

                    # 尺寸一致性惩罚：防止融合大框匹配到单人轨迹
                    track_w = track_bbox[2] - track_bbox[0]
                    track_h = track_bbox[3] - track_bbox[1]
                    det_w = det[2] - det[0]
                    det_h = det[3] - det[1]
                    track_area = max(track_w * track_h, 1.0)
                    det_area = max(det_w * det_h, 1.0)
                    area_ratio = det_area / track_area
                    # 检测框面积超过轨迹2倍时，很可能是融合框，增加惩罚
                    if area_ratio > 2.0:
                        base_cost += min(1.0, (area_ratio - 2.0) * 0.5)
                    # 检测框面积不到轨迹一半时，也不太合理
                    elif area_ratio < 0.5:
                        base_cost += min(1.0, (0.5 - area_ratio) * 0.3)

                    track_cx = (track_bbox[0] + track_bbox[2]) / 2
                    track_cy = (track_bbox[1] + track_bbox[3]) / 2
                    det_cx = (det[0] + det[2]) / 2
                    det_cy = (det[1] + det[3]) / 2

                    move_dx = det_cx - track_cx
                    move_dy = det_cy - track_cy
                    move_dist = np.sqrt(move_dx**2 + move_dy**2)

                    if len(velocity) >= 2:
                        speed = np.sqrt(velocity[0]**2 + velocity[1]**2)

                        if speed > 0.5 and move_dist > speed * 3:
                            base_cost += min(1.0, (move_dist / (speed + 1) - 3) * 0.2)

                        if speed > 0.5 and move_dist > 1:
                            vel_norm = velocity[0] / speed
                            move_norm = move_dx / move_dist

                            direction_similarity = abs(vel_norm * move_norm +
                                                      (velocity[1] / speed) * (move_dy / move_dist))

                            if direction_similarity < 0.5:
                                base_cost += 0.5 * (1 - direction_similarity)

                    cost_matrix[i, j] = base_cost

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_detections = list(detection_indices)
        unmatched_tracks = list(track_indices)

        for row, col in zip(row_indices, col_indices):
            track_idx = track_indices[row]
            det_idx = detection_indices[col]

            if cost_matrix[row, col] < 1e4:
                matched.append((track_idx, det_idx))
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)

        return matched, unmatched_detections, unmatched_tracks

    def _iou_matching(
        self,
        detections: np.ndarray,
        track_indices: List[int],
        detection_indices: List[int],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        IoU匹配（用于未确认轨迹）
        """
        if len(track_indices) == 0 or len(detection_indices) == 0:
            return [], detection_indices, track_indices

        iou_matrix = np.zeros((len(track_indices), len(detection_indices)), dtype=np.float32)

        for i, track_idx in enumerate(track_indices):
            track_bbox = self.tracks[track_idx].get_bbox()
            for j, det_idx in enumerate(detection_indices):
                iou_matrix[i, j] = self._compute_iou(track_bbox, detections[det_idx])

        cost_matrix = 1 - iou_matrix

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_detections = list(detection_indices)
        unmatched_tracks = list(track_indices)

        for row, col in zip(row_indices, col_indices):
            track_idx = track_indices[row]
            det_idx = detection_indices[col]

            if iou_matrix[row, col] >= self.iou_threshold:
                matched.append((track_idx, det_idx))
                if det_idx in unmatched_detections:
                    unmatched_detections.remove(det_idx)
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)

        return matched, unmatched_detections, unmatched_tracks

    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """计算两个边界框的IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def get_all_tracks(self) -> List[Track]:
        """获取所有轨迹"""
        return self.tracks

    def get_active_tracks(self) -> List[Track]:
        """获取活跃轨迹（已确认且最近更新）"""
        return [
            t for t in self.tracks
            if t.is_confirmed and t.time_since_update == 0
        ]

    def reset(self):
        """重置跟踪器"""
        self.tracks = []
        self.frame_count = 0
        Track.reset_count()

    def get_tracking_info(self) -> Dict[str, Any]:
        """获取跟踪统计信息"""
        confirmed = [t for t in self.tracks if t.is_confirmed]
        tentative = [t for t in self.tracks if t.is_tentative]

        return {
            "frame_count": self.frame_count,
            "total_tracks": len(self.tracks),
            "confirmed_tracks": len(confirmed),
            "tentative_tracks": len(tentative),
            "active_tracks": len([t for t in confirmed if t.time_since_update == 0]),
        }
