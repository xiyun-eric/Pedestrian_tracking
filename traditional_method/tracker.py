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

    def predict(self):
        """预测下一帧的状态"""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

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

        self.hits += 1
        self.time_since_update = 0
        self.confidence = confidence
        self.history.append(detection.copy())

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
        max_age: int = 30,
        min_hits: int = 2,
        max_iou_distance: float = 0.7,
        max_age_unmatched: int = 30,
        lambda_iou: float = 0.5,
    ):
        """
        初始化跟踪器

        Args:
            iou_threshold: IoU匹配阈值
            max_age: 轨迹最大存活帧数
            min_hits: 确认轨迹所需的最小检测次数
            max_iou_distance: 最大IoU距离
            max_age_unmatched: 未匹配轨迹最大存活帧数
            lambda_iou: IoU距离权重（马氏距离权重为1-lambda_iou）
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_iou_distance = max_iou_distance
        self.max_age_unmatched = max_age_unmatched
        self.lambda_iou = lambda_iou

        self.tracks: List[Track] = []
        self.frame_count = 0

        self._gating_threshold = 9.4877

    def update(
        self,
        detections: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        class_ids: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        更新跟踪器

        Args:
            detections: (N, 4) 边界框数组 [x1, y1, x2, y2]
            confidences: (N,) 置信度数组
            class_ids: (N,) 类别ID数组

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

        for track in self.tracks:
            track.predict()

        matched, unmatched_detections, unmatched_tracks = self._cascade_associate(
            detections, confidences
        )

        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(
                detections[det_idx],
                confidences[det_idx],
            )

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        for det_idx in unmatched_detections:
            new_track = Track(
                detections[det_idx],
                confidences[det_idx],
            )
            self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if not t.is_deleted]

        confirmed_tracks = [t for t in self.tracks if t.is_confirmed]

        return confirmed_tracks

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
