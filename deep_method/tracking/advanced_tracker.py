"""
增强版多目标跟踪器

包含以下改进：
  1. ByteTrack 二次匹配：利用低分框减少漏检
  2. 外观特征融合：IoU + 马氏距离 + ReID 相似度
  3. 社会行为约束：防止目标重叠
  4. 自适应参数调整：根据场景动态调整参数
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
import sys
from pathlib import Path

# 导入卡尔曼滤波器（从本地模块导入）
from deep_method.tracking.kalman_filter import KalmanFilter


class TrackState(Enum):
    """轨迹状态枚举"""
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


@dataclass
class AdvancedTrack:
    """
    增强版轨迹类
    
    包含外观特征、速度历史等额外信息
    """
    track_id: int
    mean: np.ndarray
    covariance: np.ndarray
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confidence: float = 1.0
    history: List[np.ndarray] = field(default_factory=list)
    feature: Optional[np.ndarray] = None
    feature_history: List[np.ndarray] = field(default_factory=list)
    velocity_history: List[np.ndarray] = field(default_factory=list)
    track_state: TrackState = TrackState.TENTATIVE
    kf: Any = None
    
    _count = 0
    
    def __init__(
        self,
        detection: np.ndarray,
        confidence: float = 1.0,
        feature: Optional[np.ndarray] = None,
        use_depth: bool = False,
        min_hits: int = 3,  # 新增：轨迹确认所需最小匹配次数
    ):
        """初始化轨迹"""
        self.track_id = AdvancedTrack._count
        AdvancedTrack._count += 1
        
        self.use_depth = use_depth
        self.min_hits = min_hits  # 存储min_hits配置
        
        # 初始化卡尔曼滤波器
        try:
            self.kf = KalmanFilter(use_depth=use_depth)
            z = KalmanFilter.bbox_to_z(detection, None)
            self.mean, self.covariance = self.kf.initiate(z)
        except:
            # 简化初始化
            self.mean = np.zeros(8)
            self.covariance = np.eye(8) * 100
            cx = (detection[0] + detection[2]) / 2
            cy = (detection[1] + detection[3]) / 2
            w = detection[2] - detection[0]
            h = detection[3] - detection[1]
            self.mean[:4] = [cx, cy, w, h]
        
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confidence = confidence
        self.history = [detection.copy()]
        self.last_seen_bbox = detection.copy()  # 最后一次匹配成功的位置（不随预测漂移）
        # 尺度信息：用于2D场景下的深度/距离判断
        det_w = detection[2] - detection[0]
        det_h = detection[3] - detection[1]
        self.last_seen_size = np.array([det_w, det_h])  # 最后匹配成功的框尺寸
        self.size_history = [self.last_seen_size.copy()]  # 尺度历史（用于判断姿态变化）
        self.feature = feature
        self.feature_history = [feature] if feature is not None else []
        self.velocity_history = []
        self.track_state = TrackState.TENTATIVE
    
    def predict(self):
        """预测下一帧状态"""
        try:
            self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        except:
            # 简化预测
            self.mean[:4] += self.mean[4:8]
        
        self.age += 1
        self.time_since_update += 1
        
        # 检查边界框是否有效
        w = self.mean[2]
        h = self.mean[3]
        if w <= 0 or h <= 0:
            self.track_state = TrackState.DELETED
    
    def update(
        self,
        detection: np.ndarray,
        confidence: float = 1.0,
        feature: Optional[np.ndarray] = None,
    ):
        """更新轨迹"""
        # 注意：不做检测框平滑（EMA），因为：
        # 1. 卡尔曼滤波器已经负责状态估计和平滑
        # 2. EMA与卡尔曼滤波器冲突，会导致框累积缩小
        # 3. 平滑后的框存入history会导致IoU匹配和尺度检查失真
        
        try:
            z = KalmanFilter.bbox_to_z(detection, None)
            self.mean, self.covariance = self.kf.update(self.mean, self.covariance, z)
        except:
            # 简化更新
            cx = (detection[0] + detection[2]) / 2
            cy = (detection[1] + detection[3]) / 2
            w = detection[2] - detection[0]
            h = detection[3] - detection[1]
            old_cx = self.mean[0]
            old_cy = self.mean[1]
            self.mean[:4] = [cx, cy, w, h]
            self.mean[4:8] = [cx - old_cx, cy - old_cy, 0, 0]
        
        self.hits += 1
        self.time_since_update = 0
        self.confidence = confidence
        self.history.append(detection.copy())
        self.last_seen_bbox = detection.copy()  # 用原始检测更新（非平滑）
        
        # 更新尺度信息
        det_w = detection[2] - detection[0]
        det_h = detection[3] - detection[1]
        self.last_seen_size = np.array([det_w, det_h])
        self.size_history.append(self.last_seen_size.copy())
        if len(self.size_history) > 30:
            self.size_history = self.size_history[-30:]
        
        # 更新特征
        if feature is not None:
            if self.feature is not None:
                # 平滑更新特征：使用较低alpha保留更多历史特征
                # 遮挡恢复后特征变化大，低alpha避免特征被完全覆盖
                alpha = 0.5
                self.feature = alpha * feature + (1 - alpha) * self.feature
                # 重新归一化
                self.feature = self.feature / (np.linalg.norm(self.feature) + 1e-8)
            else:
                self.feature = feature
            self.feature_history.append(feature)
        
        # 记录速度历史
        velocity = self.get_velocity()
        self.velocity_history.append(velocity.copy())
        if len(self.velocity_history) > 30:
            self.velocity_history = self.velocity_history[-30:]
        
        # 状态转换（使用配置的min_hits）
        if self.track_state == TrackState.TENTATIVE and self.hits >= self.min_hits:
            self.track_state = TrackState.CONFIRMED
    
    def mark_missed(self):
        """标记为丢失"""
        self.track_state = TrackState.DELETED
    
    def get_bbox(self) -> np.ndarray:
        """获取边界框 [x1, y1, x2, y2]"""
        try:
            return KalmanFilter.z_to_bbox(self.mean[:4])
        except:
            cx, cy, w, h = self.mean[:4]
            return np.array([
                cx - w / 2,
                cy - h / 2,
                cx + w / 2,
                cy + h / 2
            ], dtype=np.float32)
    
    def get_velocity(self) -> np.ndarray:
        """获取速度"""
        return self.mean[4:8]
    
    def get_predicted_bbox(self) -> np.ndarray:
        """获取预测边界框"""
        return self.get_bbox()
    
    def mahalanobis_distance(
        self,
        measurement: np.ndarray,
        only_position: bool = False,
    ) -> float:
        """计算马氏距离"""
        try:
            z = KalmanFilter.bbox_to_z(measurement, None)
            return self.kf.mahalanobis_distance(
                self.mean, self.covariance, z, only_position
            )
        except:
            # 简化距离计算
            bbox = self.get_bbox()
            center_dist = np.linalg.norm(
                np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]) -
                np.array([(measurement[0] + measurement[2]) / 2, (measurement[1] + measurement[3]) / 2])
            )
            return center_dist
    
    @property
    def is_tentative(self) -> bool:
        return self.track_state == TrackState.TENTATIVE
    
    @property
    def is_confirmed(self) -> bool:
        return self.track_state == TrackState.CONFIRMED
    
    @property
    def is_deleted(self) -> bool:
        return self.track_state == TrackState.DELETED
    
    @classmethod
    def reset_count(cls):
        cls._count = 0


@dataclass
class AdvancedTrackerConfig:
    """增强版跟踪器配置"""
    # 基础参数
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    
    # ByteTrack 参数
    use_bytetrack: bool = True
    high_conf_threshold: float = 0.5
    low_conf_threshold: float = 0.1
    
    # 外观特征参数
    use_reid: bool = True
    reid_model: str = 'osnet_x1_0'
    appearance_weight: float = 0.4
    iou_weight: float = 0.3
    mahal_weight: float = 0.3
    feature_smooth_alpha: float = 0.7
    
    # 社会行为参数
    use_social_constraint: bool = True
    social_weight: float = 0.2
    overlap_threshold: float = 0.3
    
    # 自适应参数
    use_adaptive: bool = True
    density_threshold_high: int = 20
    density_threshold_low: int = 5
    
    # 马氏距离门控
    gating_threshold: float = 9.4877
    
    # 场景参数
    scene_type: str = 'general'
    camera_motion: bool = False
    
    # 输出参数
    output_trajectory: bool = False
    output_confidence: bool = True
    
    # 类别特定参数
    class_specific_params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'pedestrian': {
            'max_age': 30,
            'iou_threshold': 0.3,
            'appearance_weight': 0.5,
        },
        'vehicle': {
            'max_age': 20,
            'iou_threshold': 0.4,
            'appearance_weight': 0.3,
        },
        'bicycle': {
            'max_age': 15,
            'iou_threshold': 0.25,
            'appearance_weight': 0.4,
        },
    })


class AdvancedTracker:
    """
    增强版多目标跟踪器
    
    改进内容：
    1. ByteTrack 二次匹配
    2. 外观特征融合
    3. 社会行为约束
    4. 自适应参数调整
    """
    
    def __init__(self, config: AdvancedTrackerConfig = None):
        self.config = config or AdvancedTrackerConfig()
        
        self.tracks: List[AdvancedTrack] = []
        self.frame_count = 0
        
        # 自适应参数（动态调整）
        self.adaptive_params = {
            'iou_threshold': self.config.iou_threshold,
            'max_age': self.config.max_age,
            'appearance_weight': self.config.appearance_weight,
        }
        
        # 场景统计
        self.scene_stats = {
            'num_detections': 0,
            'avg_density': 0,
            'motion_magnitude': 0,
        }
        
        # 特征提取器（外部传入）
        self.reid_extractor = None
    
    def set_reid_extractor(self, extractor):
        """设置 ReID 特征提取器"""
        self.reid_extractor = extractor
    
    def update(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
        features: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        low_conf_detections: Optional[np.ndarray] = None,
        low_conf_confidences: Optional[np.ndarray] = None,
    ) -> List[AdvancedTrack]:
        """
        更新跟踪器
        
        Args:
            detections: 高置信度检测 (N, 4)
            confidences: 高置信度置信度 (N,)
            features: 外观特征 (N, feature_dim)
            image: 原始图像（用于提取特征）
            low_conf_detections: 低置信度检测（ByteTrack）
            low_conf_confidences: 低置信度置信度
        
        Returns:
            确认的轨迹列表
        """
        self.frame_count += 1
        
        # 自适应参数调整
        if self.config.use_adaptive:
            self._adapt_params(len(detections))
        
        # 预测所有轨迹
        for track in self.tracks:
            track.predict()
        
        # 第一次匹配：高分框 + 所有轨迹
        matched, unmatched_dets, unmatched_tracks = self._cascade_match(
            detections, confidences, features
        )
        
        # 更新匹配的轨迹
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            det_feature = features[det_idx] if features is not None else None
            track.update(detections[det_idx], confidences[det_idx], det_feature)
        
        # ByteTrack 二次匹配：低分框 + 未匹配的确认轨迹
        if self.config.use_bytetrack and low_conf_detections is not None:
            confirmed_unmatched = [
                i for i in unmatched_tracks 
                if self.tracks[i].is_confirmed
            ]
            
            if len(confirmed_unmatched) > 0 and len(low_conf_detections) > 0:
                low_matched = self._bytetrack_second_match(
                    low_conf_detections,
                    low_conf_confidences,
                    confirmed_unmatched,
                )
                
                for track_idx, det_idx in low_matched:
                    self.tracks[track_idx].update(
                        low_conf_detections[det_idx],
                        low_conf_confidences[det_idx],
                    )
                    unmatched_tracks.remove(track_idx)
        
        # ReID 重匹配：丢失轨迹用外观特征与未匹配检测恢复
        if self.config.use_reid and features is not None and len(unmatched_dets) > 0:
            lost_tracks = [
                i for i in unmatched_tracks
                if self.tracks[i].is_confirmed
                and self.tracks[i].feature is not None
                and 1 <= self.tracks[i].time_since_update <= 30  # 遮挡30帧内可恢复（约1秒）
            ]
            
            if len(lost_tracks) > 0:
                reid_matched = self._reid_rematch(
                    detections, features, lost_tracks, unmatched_dets
                )
                for track_idx, det_idx in reid_matched:
                    self.tracks[track_idx].update(
                        detections[det_idx], confidences[det_idx],
                        features[det_idx] if features is not None else None,
                    )
                    unmatched_tracks.remove(track_idx)
                    if det_idx in unmatched_dets:
                        unmatched_dets.remove(det_idx)
        
        # 标记未匹配轨迹为丢失
        for track_idx in unmatched_tracks:
            if self.tracks[track_idx].time_since_update > self.adaptive_params['max_age']:
                self.tracks[track_idx].mark_missed()
        
        # 创建新轨迹（仅高置信度检测）
        new_track_threshold = self.config.high_conf_threshold  # 默认0.5
        for det_idx in unmatched_dets:
            # 低置信度检测不创建新轨迹，避免碎片轨迹
            if confidences[det_idx] < new_track_threshold:
                continue
            
            # 尺度+位置重叠检查：防止同一目标因姿态变化产生多个轨迹
            det = detections[det_idx]
            det_cx = (det[0] + det[2]) / 2
            det_cy = (det[1] + det[3]) / 2
            det_w = det[2] - det[0]
            det_h = det[3] - det[1]
            det_area = det_w * det_h
            
            is_duplicate = False
            for track in self.tracks:
                # 检查所有轨迹（包括TENTATIVE），防止同一目标产生多个轨迹
                # 特别重要：刚开始时轨迹都是TENTATIVE，必须检查以避免ID切换
                if track.is_deleted:
                    continue
                    
                # 检查位置重叠：检测中心在已有轨迹框内或附近
                track_bbox = track.get_bbox()
                
                # 放宽位置检查：小目标检测位置波动大，使用更宽松的范围
                # 计算轨迹框的扩展范围（根据尺寸动态调整）
                track_w = track_bbox[2] - track_bbox[0]
                track_h = track_bbox[3] - track_bbox[1]
                track_area = track_w * track_h
                
                # 小目标扩展范围更大（检测位置波动大）
                if track_area < 500:  # 小目标
                    expand_ratio = 0.3  # 扩展30%
                elif track_area < 2000:  # 中目标
                    expand_ratio = 0.2  # 扩展20%
                else:  # 大目标
                    expand_ratio = 0.1  # 扩展10%
                
                expanded_bbox = [
                    track_bbox[0] - track_w * expand_ratio,
                    track_bbox[1] - track_h * expand_ratio,
                    track_bbox[2] + track_w * expand_ratio,
                    track_bbox[3] + track_h * expand_ratio,
                ]
                
                if (expanded_bbox[0] <= det_cx <= expanded_bbox[2] and
                    expanded_bbox[1] <= det_cy <= expanded_bbox[3]):
                    # 位置重叠，再检查尺度是否相近
                    if len(track.size_history) > 0:
                        hist_areas = [s[0] * s[1] for s in track.size_history[-5:]]
                        avg_area = np.mean(hist_areas)
                        if avg_area > 0 and det_area > 0:
                            area_ratio = det_area / avg_area
                            
                            # 根据目标尺寸动态调整尺度阈值
                            # 小目标检测波动大，放宽阈值
                            if avg_area < 500:  # 小目标
                                scale_min, scale_max = 0.3, 3.0
                            elif avg_area < 2000:  # 中目标
                                scale_min, scale_max = 0.4, 2.5
                            else:  # 大目标
                                scale_min, scale_max = 0.5, 2.0
                            
                            if scale_min < area_ratio < scale_max:
                                is_duplicate = True
                                break
            
            if is_duplicate:
                continue
            
            det_feature = features[det_idx] if features is not None else None
            new_track = AdvancedTrack(
                detections[det_idx],
                confidences[det_idx],
                det_feature,
                min_hits=self.config.min_hits,  # 使用配置的min_hits
            )
            self.tracks.append(new_track)
        
        # 清理已删除轨迹
        self.tracks = [t for t in self.tracks if not t.is_deleted]
        
        # 返回确认轨迹
        return [t for t in self.tracks if t.is_confirmed]
    
    def _cascade_match(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
        features: Optional[np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        级联匹配
        
        按轨迹的 time_since_update 从小到大匹配
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        confirmed = [i for i, t in enumerate(self.tracks) if t.is_confirmed]
        tentative = [i for i, t in enumerate(self.tracks) if t.is_tentative]
        
        matched = []
        unmatched_dets = list(range(len(detections)))
        
        # 按年龄级联匹配确认轨迹
        for age in range(1, self.config.max_age + 1):
            if len(unmatched_dets) == 0:
                break
            
            track_indices = [
                i for i in confirmed
                if self.tracks[i].time_since_update == age
            ]
            
            if len(track_indices) == 0:
                continue
            
            # 计算代价矩阵
            cost_matrix = self._compute_cost_matrix(
                detections, confidences, features, track_indices, unmatched_dets
            )
            
            # 匈牙利算法
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 收集本轮匹配
            current_matched = []
            for row, col in zip(row_ind, col_ind):
                # 代价阈值过滤：根据目标尺寸动态调整阈值
                # 小目标（中远处）使用更高阈值（更宽容）
                track_idx = track_indices[row]
                track = self.tracks[track_idx]
                
                # 根据轨迹历史面积确定阈值
                if len(track.size_history) > 0:
                    hist_areas = [s[0] * s[1] for s in track.size_history[-5:]]
                    avg_area = np.mean(hist_areas)
                    
                    if avg_area < 500:  # 小目标（中远处）
                        cost_threshold = 0.7  # 更宽容
                    elif avg_area < 2000:  # 中目标
                        cost_threshold = 0.5  # 标准
                    else:  # 大目标（近处）
                        cost_threshold = 0.4  # 更严格
                else:
                    cost_threshold = 0.5  # 默认
                
                if cost_matrix[row, col] < cost_threshold:
                    det_idx = unmatched_dets[col]
                    current_matched.append((track_idx, det_idx))
            
            # 更新匹配列表和未匹配检测
            matched.extend(current_matched)
            for _, det_idx in current_matched:
                if det_idx in unmatched_dets:
                    unmatched_dets.remove(det_idx)
        
        # IoU 匹配暂定轨迹
        if len(tentative) > 0 and len(unmatched_dets) > 0:
            iou_matched = self._iou_match(detections, tentative, unmatched_dets)
            matched.extend(iou_matched)
        
        unmatched_tracks = [
            i for i in range(len(self.tracks))
            if i not in [m[0] for m in matched]
        ]
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _compute_cost_matrix(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
        features: Optional[np.ndarray],
        track_indices: List[int],
        det_indices: List[int],
    ) -> np.ndarray:
        """
        计算融合代价矩阵
        
        IoU + 马氏距离 + 外观相似度 + 社会行为约束
        """
        cost_matrix = np.zeros((len(track_indices), len(det_indices)))
        
        for i, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            track_bbox = track.get_bbox()
            
            for j, det_idx in enumerate(det_indices):
                det = detections[det_idx]
                
                # 1. IoU 距离
                iou = self._compute_iou(track_bbox, det)
                iou_dist = 1 - iou
                
                # 2. 马氏距离
                mahal_dist = track.mahalanobis_distance(det)
                mahal_norm = mahal_dist / self.config.gating_threshold
                
                # 3. 外观相似度
                appearance_dist = 0.5  # 默认值
                if features is not None and track.feature is not None:
                    det_feature = features[det_idx]
                    similarity = np.dot(track.feature, det_feature) / (
                        np.linalg.norm(track.feature) * np.linalg.norm(det_feature) + 1e-8
                    )
                    appearance_dist = 1 - similarity
                
                # 门控检查
                if mahal_dist > self.config.gating_threshold:
                    cost_matrix[i, j] = 1e5
                    continue
                
                # 5. 尺度一致性代价（对小目标更宽容）
                # 原理：2D场景下，同一人的检测框面积不应剧烈变化
                # 但小目标（中远处）的检测框大小波动更大，需要放宽阈值
                scale_cost = 0
                det_w = det[2] - det[0]
                det_h = det[3] - det[1]
                det_area = det_w * det_h
                if det_area > 0 and len(track.size_history) > 0:
                    # 使用历史平均面积（更稳定）
                    hist_areas = [s[0] * s[1] for s in track.size_history[-5:]]
                    avg_area = np.mean(hist_areas)
                    if avg_area > 0:
                        area_ratio = det_area / avg_area
                        
                        # 根据目标尺寸动态调整阈值
                        # 小目标（面积<500像素²）检测波动大，放宽阈值
                        # 中目标（500~2000像素²）使用标准阈值
                        # 大目标（>2000像素²）使用严格阈值
                        if avg_area < 500:  # 小目标（中远处）
                            # 放宽阈值：面积比在0.3~3.0范围内不惩罚
                            if area_ratio > 3.0:
                                scale_cost = min(1.0, (area_ratio - 3.0) * 0.3)
                            elif area_ratio < 0.3:
                                scale_cost = min(1.0, (0.3 - area_ratio) * 0.5)
                            elif area_ratio > 2.0 or area_ratio < 0.5:
                                # 轻微惩罚
                                scale_cost = 0.05 * abs(area_ratio - 1.0)
                        elif avg_area < 2000:  # 中目标
                            # 标准阈值：面积比在0.5~2.0范围内不惩罚
                            if area_ratio > 2.0:
                                scale_cost = min(1.0, (area_ratio - 2.0) * 0.5)
                            elif area_ratio < 0.5:
                                scale_cost = min(1.0, (0.5 - area_ratio) * 1.0)
                            elif area_ratio > 1.5 or area_ratio < 0.67:
                                scale_cost = 0.1 * abs(area_ratio - 1.0)
                        else:  # 大目标（近处）
                            # 严格阈值：面积比在0.6~1.7范围内不惩罚
                            if area_ratio > 1.7:
                                scale_cost = min(1.0, (area_ratio - 1.7) * 0.7)
                            elif area_ratio < 0.6:
                                scale_cost = min(1.0, (0.6 - area_ratio) * 1.5)
                            elif area_ratio > 1.3 or area_ratio < 0.8:
                                scale_cost = 0.15 * abs(area_ratio - 1.0)
                
                # 4. 社会行为约束
                social_cost = 0
                if self.config.use_social_constraint:
                    social_cost = self._compute_social_cost(track_idx, det)
                
                # 动态调整外观权重：IoU高时降低外观权重
                # 原理：当空间位置高度重叠时，即使外观变化（如背身/侧身），
                # 也应信任空间匹配而非外观匹配
                base_app_w = self.adaptive_params.get('appearance_weight', self.config.appearance_weight)
                iou_w = self.adaptive_params.get('iou_weight', self.config.iou_weight)
                
                # 对于刚匹配过的轨迹（time_since_update小），更激进地信任空间位置
                # 因为短时间内同一位置的目标大概率是同一个人
                if track.time_since_update <= 1:
                    # 刚匹配或连续匹配：IoU>0.2就大幅降低外观权重
                    if iou > 0.2:
                        dynamic_app_w = base_app_w * 0.1
                        dynamic_iou_w = iou_w + base_app_w * 0.9
                    elif iou > 0.1:
                        ratio = (iou - 0.1) / 0.1
                        dynamic_app_w = base_app_w * (1 - 0.8 * ratio)
                        dynamic_iou_w = iou_w + base_app_w * 0.8 * ratio
                    else:
                        dynamic_app_w = base_app_w
                        dynamic_iou_w = iou_w
                else:
                    # 丢失过一段时间的轨迹：保守一些
                    if iou > 0.5:
                        dynamic_app_w = base_app_w * 0.2
                        dynamic_iou_w = iou_w + base_app_w * 0.8
                    elif iou > 0.3:
                        ratio = (iou - 0.3) / 0.2
                        dynamic_app_w = base_app_w * (1 - 0.6 * ratio)
                        dynamic_iou_w = iou_w + base_app_w * 0.6 * ratio
                    else:
                        dynamic_app_w = base_app_w
                        dynamic_iou_w = iou_w
                
                # 融合代价
                cost = (
                    dynamic_iou_w * iou_dist +
                    self.config.mahal_weight * mahal_norm +
                    dynamic_app_w * appearance_dist +
                    self.config.social_weight * social_cost +
                    0.3 * scale_cost  # 尺度一致性代价
                )
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _compute_social_cost(
        self,
        track_idx: int,
        det: np.ndarray,
    ) -> float:
        """
        计算社会行为代价
        
        防止目标重叠
        """
        cost = 0
        
        for other_track in self.tracks:
            if other_track.track_id == self.tracks[track_idx].track_id:
                continue
            
            other_bbox = other_track.get_predicted_bbox()
            predicted_iou = self._compute_iou(det, other_bbox)
            
            if predicted_iou > self.config.overlap_threshold:
                cost += 0.5 * predicted_iou
        
        return cost
    
    def _bytetrack_second_match(
        self,
        low_detections: np.ndarray,
        low_confidences: np.ndarray,
        track_indices: List[int],
    ) -> List[Tuple[int, int]]:
        """
        ByteTrack 二次匹配
        
        低分框与未匹配的确认轨迹进行 IoU 匹配
        """
        matched = []
        
        for track_idx in track_indices:
            track_bbox = self.tracks[track_idx].get_bbox()
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(low_detections):
                iou = self._compute_iou(track_bbox, det)
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                matched.append((track_idx, best_det_idx))
        
        return matched
    
    def _reid_rematch(
        self,
        detections: np.ndarray,
        features: np.ndarray,
        track_indices: List[int],
        det_indices: List[int],
    ) -> List[Tuple[int, int]]:
        """
        ReID 重匹配：用外观特征恢复遮挡后的丢失轨迹
        
        关键改进：
        1. 使用 last_seen_bbox（最后已知位置）而非预测位置
        2. 用中心点距离替代IoU，扩大搜索范围
        3. 运动一致性约束：检测的运动方向应与轨迹历史运动一致
        4. 使用匈牙利算法全局最优匹配，避免贪心法导致的ID混淆
        """
        if len(track_indices) == 0 or len(det_indices) == 0:
            return []
        
        # 构建代价矩阵
        cost_matrix = np.full((len(track_indices), len(det_indices)), 1e5)
        
        for i, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            if track.feature is None:
                continue
            
            last_bbox = track.last_seen_bbox
            last_cx = (last_bbox[0] + last_bbox[2]) / 2
            last_cy = (last_bbox[1] + last_bbox[3]) / 2
            last_w = last_bbox[2] - last_bbox[0]
            last_h = last_bbox[3] - last_bbox[1]
            
            # 搜索半径：基于目标大小和丢失时间
            search_radius = max(last_w, last_h) * (1.0 + 0.3 * track.time_since_update)
            search_radius = min(search_radius, 300)
            
            # 计算轨迹的历史运动方向（用于运动一致性检查）
            last_velocity = np.array([0.0, 0.0])
            if len(track.velocity_history) > 0:
                # 取最近几帧的平均速度
                recent_vel = track.velocity_history[-3:]
                for v in recent_vel:
                    last_velocity[0] += v[0]
                    last_velocity[1] += v[1]
                last_velocity /= len(recent_vel)
            
            for j, det_idx in enumerate(det_indices):
                det_feature = features[det_idx]
                det = detections[det_idx]
                
                # 1. 外观相似度
                similarity = np.dot(track.feature, det_feature) / (
                    np.linalg.norm(track.feature) * np.linalg.norm(det_feature) + 1e-8
                )
                
                # 2. 中心点距离（用最后已知位置）
                det_cx = (det[0] + det[2]) / 2
                det_cy = (det[1] + det[3]) / 2
                center_dist = np.sqrt((det_cx - last_cx)**2 + (det_cy - last_cy)**2)
                
                # 超出搜索范围直接跳过
                if center_dist > search_radius * 1.5:
                    continue
                
                # 3. 运动一致性检查
                # 检测位置相对于最后已知位置的位移方向
                displacement = np.array([det_cx - last_cx, det_cy - last_cy])
                motion_consistent = True
                motion_penalty = 0
                
                if np.linalg.norm(last_velocity) > 2.0 and np.linalg.norm(displacement) > 5.0:
                    # 轨迹有明显运动且位移较大时，检查方向一致性
                    vel_dir = last_velocity / (np.linalg.norm(last_velocity) + 1e-8)
                    disp_dir = displacement / (np.linalg.norm(displacement) + 1e-8)
                    cos_angle = np.dot(vel_dir, disp_dir)
                    
                    if cos_angle < -0.3:
                        # 位移方向与历史运动方向相反（如：静止目标被移动检测匹配）
                        motion_consistent = False
                        motion_penalty = 0.5
                    elif cos_angle < 0.3:
                        # 方向不太一致，轻微惩罚
                        motion_penalty = 0.2
                
                # 4. IoU（用最后已知位置）
                iou = self._compute_iou(last_bbox, det)
                
                # 5. 空间可信度
                spatial_confidence = max(0, 1 - center_dist / search_radius)
                
                # 6. 尺度一致性检查
                # 遮挡恢复时，目标不应突然变大/变小（近处目标不会变成远处目标）
                scale_penalty = 0
                det_w = det[2] - det[0]
                det_h = det[3] - det[1]
                det_area = det_w * det_h
                if det_area > 0 and len(track.size_history) > 0:
                    hist_areas = [s[0] * s[1] for s in track.size_history[-5:]]
                    avg_area = np.mean(hist_areas)
                    if avg_area > 0:
                        area_ratio = det_area / avg_area
                        if area_ratio > 2.0:
                            # 检测框远大于历史：近处目标不可能变成远处目标
                            scale_penalty = min(1.0, (area_ratio - 2.0) * 0.8)
                        elif area_ratio < 0.5:
                            # 检测框远小于历史：远处目标不可能变成近处目标
                            scale_penalty = min(1.0, (0.5 - area_ratio) * 1.5)
                
                # 匹配条件（满足其一即可，但运动不一致或尺度不一致时更严格）
                is_match = False
                if not motion_consistent:
                    # 运动不一致：要求更高的外观相似度
                    if similarity > 0.85 and center_dist < search_radius * 0.5 and scale_penalty < 0.3:
                        is_match = True
                elif scale_penalty > 0.3:
                    # 尺度差异大：要求更高的外观相似度和空间可信度
                    if similarity > 0.85 and spatial_confidence > 0.7:
                        is_match = True
                else:
                    if similarity > 0.6 and center_dist < search_radius:
                        is_match = True
                    elif similarity > 0.8 and center_dist < search_radius * 1.5:
                        is_match = True
                    elif iou > 0.2 and similarity > 0.4:
                        is_match = True
                
                if is_match:
                    # 综合代价：外观 + 空间 + 运动惩罚 + 尺度惩罚
                    cost = (1 - similarity) * 0.4 + (1 - spatial_confidence) * 0.25 + motion_penalty * 0.2 + scale_penalty * 0.15
                    cost_matrix[i, j] = cost
        
        # 使用匈牙利算法全局最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched = []
        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] < 0.8:  # 代价阈值
                matched.append((track_indices[row], det_indices[col]))
        
        return matched
    
    def _iou_match(
        self,
        detections: np.ndarray,
        track_indices: List[int],
        det_indices: List[int],
    ) -> List[Tuple[int, int]]:
        """IoU 匹配"""
        matched = []
        
        iou_matrix = np.zeros((len(track_indices), len(det_indices)))
        for i, track_idx in enumerate(track_indices):
            track_bbox = self.tracks[track_idx].get_bbox()
            for j, det_idx in enumerate(det_indices):
                iou_matrix[i, j] = self._compute_iou(track_bbox, detections[det_idx])
        
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for row, col in zip(row_ind, col_ind):
            # IoU阈值：使用配置的iou_threshold
            if iou_matrix[row, col] >= self.config.iou_threshold:
                matched.append((track_indices[row], det_indices[col]))
        
        return matched
    
    def _adapt_params(self, num_detections: int):
        """
        自适应参数调整
        
        根据目标密度动态调整参数
        """
        # 更新场景统计
        self.scene_stats['num_detections'] = num_detections
        
        # 密集场景
        if num_detections > self.config.density_threshold_high:
            self.adaptive_params['iou_threshold'] = 0.4
            self.adaptive_params['appearance_weight'] = 0.5
            self.adaptive_params['iou_weight'] = 0.2
        
        # 稀疏场景
        elif num_detections < self.config.density_threshold_low:
            self.adaptive_params['iou_threshold'] = 0.2
            self.adaptive_params['appearance_weight'] = 0.3
            self.adaptive_params['iou_weight'] = 0.5
        
        else:
            self.adaptive_params['iou_threshold'] = self.config.iou_threshold
            self.adaptive_params['appearance_weight'] = self.config.appearance_weight
            self.adaptive_params['iou_weight'] = self.config.iou_weight
    
    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """计算 IoU"""
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
    
    def reset(self):
        """重置跟踪器"""
        self.tracks = []
        self.frame_count = 0
        AdvancedTrack.reset_count()
    
    def get_tracking_info(self) -> Dict:
        """获取跟踪信息"""
        return {
            'frame_count': self.frame_count,
            'total_tracks': len(self.tracks),
            'confirmed_tracks': len([t for t in self.tracks if t.is_confirmed]),
            'adaptive_params': self.adaptive_params,
        }