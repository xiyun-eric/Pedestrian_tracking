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
        # 锚点特征：轨迹确认时保存的原始特征，不随匹配更新
        # 用于检测特征漂移（错误匹配导致特征逐渐偏离原始身份）
        self.anchor_feature = feature.copy() if feature is not None else None
        self.anchor_set = False  # 是否已设置锚点（确认时设置）
        self.velocity_history = []
        self.velocity_ema = np.zeros(2, dtype=np.float32)  # EMA平滑速度（仅cx,cy方向）
        self.velocity_ema_initialized = False  # EMA是否已初始化

        # 伪三维速度：用尺度变化率模拟深度方向运动
        # scale_ratio > 1 表示目标变大（靠近相机），< 1 表示目标变小（远离相机）
        self.scale_ratio_ema = 1.0  # EMA平滑的尺度变化率
        self.scale_ratio_ema_initialized = False
        self.pseudo_3d_velocity = np.zeros(3, dtype=np.float32)  # [vx, vy, vz] 伪三维速度
        self.pseudo_3d_velocity_initialized = False

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
        # 关键防护：外观距离大时不更新特征，防止特征漂移
        if feature is not None:
            should_update_feature = True
            if self.feature is not None:
                # 计算新特征与当前特征的距离
                sim = np.dot(self.feature, feature) / (
                    np.linalg.norm(self.feature) * np.linalg.norm(feature) + 1e-8
                )
                app_dist = 1 - sim
                if app_dist > 0.3:
                    # 外观距离>0.3（相似度<0.7），可能是错误匹配，不更新特征
                    should_update_feature = False

            if should_update_feature:
                if self.feature is not None:
                    # 平滑更新特征：使用较低alpha保留更多历史特征
                    alpha = 0.5
                    self.feature = alpha * feature + (1 - alpha) * self.feature
                    self.feature = self.feature / (np.linalg.norm(self.feature) + 1e-8)
                else:
                    self.feature = feature
            self.feature_history.append(feature)
        
        # 记录速度历史
        velocity = self.get_velocity()
        self.velocity_history.append(velocity.copy())
        if len(self.velocity_history) > 30:
            self.velocity_history = self.velocity_history[-30:]
        
        # 更新EMA速度（仅cx, cy方向分量）
        current_vel = velocity[:2].copy()  # vx, vy
        if not self.velocity_ema_initialized:
            self.velocity_ema = current_vel.astype(np.float32)
            self.velocity_ema_initialized = True
        else:
            alpha = 0.4  # EMA平滑系数（新速度权重）
            self.velocity_ema = alpha * current_vel.astype(np.float32) + (1 - alpha) * self.velocity_ema

        # 更新伪三维速度：尺度变化率 → 深度方向速度
        # scale_ratio = 当前面积 / 历史面积，>1 表示靠近相机，<1 表示远离
        if len(self.size_history) >= 2:
            prev_area = self.size_history[-2][0] * self.size_history[-2][1]
            curr_area = self.last_seen_size[0] * self.last_seen_size[1]
            if prev_area > 1e-4:
                scale_ratio = curr_area / prev_area
            else:
                scale_ratio = 1.0

            if not self.scale_ratio_ema_initialized:
                self.scale_ratio_ema = scale_ratio
                self.scale_ratio_ema_initialized = True
            else:
                sr_alpha = 0.4
                self.scale_ratio_ema = sr_alpha * scale_ratio + (1 - sr_alpha) * self.scale_ratio_ema

            # 伪三维速度：[vx, vy, vz]
            # vz 用尺度变化率偏移量近似：scale_ratio > 1 → vz > 0（靠近），< 1 → vz < 0（远离）
            vz = (self.scale_ratio_ema - 1.0) * np.sqrt(curr_area + 1e-4)
            self.pseudo_3d_velocity = np.array([
                self.velocity_ema[0],
                self.velocity_ema[1],
                vz
            ], dtype=np.float32)
            self.pseudo_3d_velocity_initialized = True
        
        # 状态转换（使用配置的min_hits）
        if self.track_state == TrackState.TENTATIVE and self.hits >= self.min_hits:
            self.track_state = TrackState.CONFIRMED
            # 确认时设置锚点特征（不随后续匹配更新）
            if self.feature is not None and not self.anchor_set:
                self.anchor_feature = self.feature.copy()
                self.anchor_set = True
    
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
    appearance_weight: float = 0.5
    iou_weight: float = 0.2
    mahal_weight: float = 0.15
    feature_smooth_alpha: float = 0.7
    
    # 社会行为参数
    use_social_constraint: bool = True
    social_weight: float = 0.1
    overlap_threshold: float = 0.3
    
    # 自适应参数
    use_adaptive: bool = True
    density_threshold_high: int = 20
    density_threshold_low: int = 5
    
    # 马氏距离门控
    gating_threshold: float = 9.4877

    # 运动一致性约束（含伪三维扩展）
    motion_consistency_weight: float = 0.15
    """运动一致性代价权重（辅助约束，外观为主时运动仅作辅助验证）"""
    motion_speed_threshold: float = 1.0
    """静止目标速度阈值（像素/帧），低于此值视为静止"""
    motion_displacement_threshold: float = 2.0
    """最小位移阈值（像素），低于此值不检查方向"""
    motion_ema_alpha: float = 0.4
    """速度EMA平滑系数"""

    # 伪三维深度约束（运动一致性的深度扩展，非独立代价）
    pseudo_3d_scale_threshold: float = 0.02
    """尺度变化率阈值，低于此值视为无深度方向运动，退化为纯2D检查"""
    pseudo_3d_direction_weight: float = 0.5
    """伪三维方向一致性在3D代价中的权重（0~1，剩余给速度一致性）"""
    pseudo_3d_new_track_frames: int = 3
    """新轨迹（hits <= 此值）不检查伪三维深度约束"""
    pseudo_3d_lost_frames: int = 5
    """丢失超过此帧数后，深度约束大幅降权"""
    pseudo_3d_depth_cost_weight: float = 0.3
    """深度方向矛盾在3D代价中的权重"""

    # 尺度一致性约束
    scale_weight: float = 0.15
    """尺度一致性代价权重（辅助约束），设为0禁用"""

    # 非线性惩罚参数（尺度、空间位置、速度变化异常时非线性惩罚）
    nonlinear_penalty_enabled: bool = True
    """是否启用非线性惩罚（变化越大惩罚越大）"""
    nonlinear_gentle_slope: float = 0.3
    """非线性惩罚中等区间斜率（二次项系数，控制温和区间增长速度）"""
    nonlinear_steep_slope: float = 1.5
    """非线性惩罚大偏差区间斜率（指数项系数，控制大偏差惩罚陡峭程度）"""
    nonlinear_transition: float = 1.0
    """非线性惩罚从二次到指数的过渡点（归一化偏差值）"""

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
        # 检测保护机制：记录每轮中与稳定轨迹"接近匹配"的检测
        # 阻止这些检测在后续轮次中被丢失更久的轨迹抢走
        protected_dets = {}  # det_idx -> (track_idx, cost, appearance_dist)

        # 交叉外观检查：记录每个检测与已处理稳定轨迹的最佳外观匹配
        # 仅记录 age=1 轮（最稳定的轨迹）的外观信息
        best_appearance_match = {}  # det_idx -> appearance_dist

        for age in range(1, self.config.max_age + 1):
            if len(unmatched_dets) == 0:
                break
            
            track_indices = [
                i for i in confirmed
                if self.tracks[i].time_since_update == age
            ]
            
            if len(track_indices) == 0:
                continue
            
            # 过滤掉被保护的检测（已被更稳定轨迹声明关联）
            available_dets = [d for d in unmatched_dets if d not in protected_dets]
            if len(available_dets) == 0:
                continue
            
            # 计算代价矩阵
            cost_matrix = self._compute_cost_matrix(
                detections, confidences, features, track_indices, available_dets
            )
            
            # 匈牙利算法
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 收集本轮匹配
            current_matched = []
            for row, col in zip(row_ind, col_ind):
                # 代价阈值过滤：根据目标尺寸动态调整阈值
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
                
                cost_val = cost_matrix[row, col]
                det_idx = available_dets[col]

                if cost_val < cost_threshold:
                    # 交叉外观检查：仅对丢失轨迹（age>1）生效
                    # 如果该检测与某个稳定轨迹的外观更匹配，阻止丢失轨迹抢走它
                    if age > 1 and det_idx in best_appearance_match and features is not None:
                        if track.feature is not None:
                            det_feature = features[det_idx]
                            cur_app_dist = 1 - np.dot(track.feature, det_feature) / (
                                np.linalg.norm(track.feature) * np.linalg.norm(det_feature) + 1e-8
                            )
                            stable_app_dist = best_appearance_match[det_idx]
                            # 稳定轨迹外观更匹配（差距 > 0.2），拒绝丢失轨迹的匹配
                            if stable_app_dist < cur_app_dist - 0.2:
                                continue

                    current_matched.append((track_idx, det_idx))
                    # 匹配成功，移除该检测的保护标记
                    if det_idx in protected_dets:
                        del protected_dets[det_idx]
                elif age == 1 and cost_val < cost_threshold * 1.5:
                    # 仅稳定轨迹（age=1）可以声明检测保护
                    # 代价接近但未通过阈值，声明关联防止被丢失轨迹抢走
                    if det_idx not in protected_dets:
                        # 记录外观距离用于后续交叉检查
                        app_d = 0.5
                        if features is not None and track.feature is not None:
                            det_feature = features[det_idx]
                            app_d = 1 - np.dot(track.feature, det_feature) / (
                                np.linalg.norm(track.feature) * np.linalg.norm(det_feature) + 1e-8
                            )
                        protected_dets[det_idx] = (track_idx, cost_val, app_d)

            # age=1轮结束后，记录每个检测与稳定轨迹的最佳外观匹配
            # 这些信息用于后续轮次的交叉外观检查
            if age == 1 and features is not None:
                for track_idx in track_indices:
                    track = self.tracks[track_idx]
                    if track.feature is None:
                        continue
                    for det_idx in available_dets:
                        det_feature = features[det_idx]
                        app_dist = 1 - np.dot(track.feature, det_feature) / (
                            np.linalg.norm(track.feature) * np.linalg.norm(det_feature) + 1e-8
                        )
                        if det_idx not in best_appearance_match or app_dist < best_appearance_match[det_idx]:
                            best_appearance_match[det_idx] = app_dist
            
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
                anchor_dist = None  # 锚点特征与检测的距离
                if features is not None and track.feature is not None:
                    det_feature = features[det_idx]
                    similarity = np.dot(track.feature, det_feature) / (
                        np.linalg.norm(track.feature) * np.linalg.norm(det_feature) + 1e-8
                    )
                    appearance_dist = 1 - similarity
                    # 锚点距离：检测与轨迹原始身份特征的差距
                    if track.anchor_feature is not None:
                        anchor_sim = np.dot(track.anchor_feature, det_feature) / (
                            np.linalg.norm(track.anchor_feature) * np.linalg.norm(det_feature) + 1e-8
                        )
                        anchor_dist = 1 - anchor_sim
                
                # 门控检查
                if mahal_dist > self.config.gating_threshold:
                    cost_matrix[i, j] = 1e5
                    continue
                
                # 4. 社会行为约束
                social_cost = 0
                if self.config.use_social_constraint:
                    social_cost = self._compute_social_cost(track_idx, det)
                
                # 5. 尺度一致性代价（对小目标更宽容）
                # 负责"面积是否合理"的静态检查，与运动约束中的深度方向检查互补
                scale_cost = self._compute_scale_cost(track, det)
                
                # 6. 统一运动一致性约束（2D方向/速度 + 伪三维深度扩展）
                motion_cost = 0
                if self.config.motion_consistency_weight > 0:
                    motion_cost = self._compute_motion_consistency_cost(
                        track, det, appearance_dist
                    )
                
                # 动态调整外观权重
                # 核心原则：外观降权仅在"外观距离较小（可能是视角变化）"时生效
                #   目的：避免同一人正侧身导致的ID切换
                #   约束：外观明显不同时绝不降权，防止不同人交汇时ID切换
                #
                #   - IoU高 + 外观相似（app_dist<0.25）：同一人视角微变，大幅降权外观，信任空间
                #   - IoU高 + 外观中等差异（0.25<app_dist<0.45）：可能是转身，适度降权外观
                #   - IoU高 + 外观差异大（app_dist>0.45）：不同人交汇，保持完整外观权重，不降权
                #   - IoU低：空间位置无参考价值，外观为主
                #   - 锚点漂移：当前特征已偏离原始身份，用锚点距离替代外观距离判断
                base_app_w = self.adaptive_params.get('appearance_weight', self.config.appearance_weight)
                iou_w = self.adaptive_params.get('iou_weight', self.config.iou_weight)
                has_reid = features is not None and track.feature is not None

                # 锚点漂移检测：如果当前特征与锚点距离大，说明特征已漂移
                # 此时用锚点距离替代外观距离做判断更可靠
                feature_drifted = False
                effective_app_dist = appearance_dist
                if anchor_dist is not None and anchor_dist > 0.35:
                    # 检测与锚点（原始身份）差距大 → 可能不是同一人
                    feature_drifted = True
                    effective_app_dist = max(appearance_dist, anchor_dist)

                # 外观距离超过阈值时，不进行外观降权（外观明显不同≠视角变化）
                # 只有外观距离较小时，才可能是同一人的视角变化，允许降权
                APP_DIST_VIEWPOINT_THRESHOLD = 0.45  # 超过此值视为不同人，不降权

                if effective_app_dist >= APP_DIST_VIEWPOINT_THRESHOLD:
                    # 外观明显不同：不可能是视角变化，保持完整外观权重
                    dynamic_app_w = base_app_w
                    dynamic_iou_w = iou_w
                elif track.time_since_update <= 1:
                    if iou > 0.5:
                        if not has_reid:
                            dynamic_app_w = base_app_w * 0.15
                            dynamic_iou_w = iou_w + base_app_w * 0.85
                        elif effective_app_dist < 0.25:
                            dynamic_app_w = base_app_w * 0.15
                            dynamic_iou_w = iou_w + base_app_w * 0.85
                        else:
                            # 0.25 < app_dist < 0.45：可能是转身，适度降权
                            dynamic_app_w = base_app_w * 0.5
                            dynamic_iou_w = iou_w + base_app_w * 0.5
                    elif iou > 0.3:
                        if not has_reid:
                            dynamic_app_w = base_app_w * 0.5
                            dynamic_iou_w = iou_w + base_app_w * 0.5
                        elif effective_app_dist < 0.25:
                            dynamic_app_w = base_app_w * 0.4
                            dynamic_iou_w = iou_w + base_app_w * 0.6
                        else:
                            dynamic_app_w = base_app_w * 0.7
                            dynamic_iou_w = iou_w + base_app_w * 0.3
                    else:
                        dynamic_app_w = base_app_w
                        dynamic_iou_w = iou_w
                else:
                    if iou > 0.5:
                        if not has_reid:
                            dynamic_app_w = base_app_w * 0.3
                            dynamic_iou_w = iou_w + base_app_w * 0.7
                        elif effective_app_dist < 0.25:
                            dynamic_app_w = base_app_w * 0.3
                            dynamic_iou_w = iou_w + base_app_w * 0.7
                        else:
                            dynamic_app_w = base_app_w * 0.6
                            dynamic_iou_w = iou_w + base_app_w * 0.4
                    elif iou > 0.3:
                        if not has_reid:
                            dynamic_app_w = base_app_w * 0.7
                            dynamic_iou_w = iou_w + base_app_w * 0.3
                        elif effective_app_dist < 0.25:
                            dynamic_app_w = base_app_w * 0.6
                            dynamic_iou_w = iou_w + base_app_w * 0.4
                        else:
                            dynamic_app_w = base_app_w * 0.8
                            dynamic_iou_w = iou_w + base_app_w * 0.2
                    else:
                        dynamic_app_w = base_app_w
                        dynamic_iou_w = iou_w

                # 外观差距极大时的额外惩罚
                # 即使IoU高，如果外观完全不像，也应增加代价防止ID互换
                appearance_penalty = 0.0
                if has_reid and appearance_dist > 0.6:
                    appearance_penalty = (appearance_dist - 0.6) * 0.5

                # 锚点漂移惩罚：检测与锚点（原始身份）差距大时额外惩罚
                # 防止特征漂移后"自我强化"导致持续错误匹配
                anchor_penalty = 0.0
                if anchor_dist is not None and anchor_dist > 0.35:
                    anchor_penalty = (anchor_dist - 0.35) * 0.6

                # 空间距离惩罚：基于轨迹最后已知位置与检测的距离
                # 防止外观相似但空间距离远的错误匹配（如穿相似衣服的不同人）
                # 使用最后已知位置而非预测位置，因为预测位置可能漂移
                # 非线性惩罚：距离越远惩罚越陡峭
                spatial_penalty = 0.0
                if track.time_since_update > 0 and has_reid:
                    # 用轨迹最后已知位置（last_seen_bbox）计算距离
                    last_bbox = track.last_seen_bbox
                    if last_bbox is not None:
                        last_cx = (last_bbox[0] + last_bbox[2]) / 2
                        last_cy = (last_bbox[1] + last_bbox[3]) / 2
                        det_cx = (det[0] + det[2]) / 2
                        det_cy = (det[1] + det[3]) / 2
                        # 归一化距离：用轨迹尺寸作为参考尺度
                        last_h = last_bbox[3] - last_bbox[1]
                        if last_h > 1e-4:
                            dist_pixels = np.sqrt((det_cx - last_cx)**2 + (det_cy - last_cy)**2)
                            dist_normalized = dist_pixels / last_h  # 以身高为单位的距离
                            # 丢失时间越长，允许的距离越大（目标可能移动了）
                            max_allowed = 1.0 + 0.3 * track.time_since_update
                            if dist_normalized > max_allowed:
                                excess = dist_normalized - max_allowed
                                # 非线性惩罚：超出允许距离越多，惩罚越陡
                                spatial_penalty = self._nonlinear_penalty(
                                    deviation=dist_normalized,
                                    threshold=max_allowed,
                                    max_penalty=0.5,
                                )
                
                # 统一运动约束权重调整（合并2D+3D的边界处理）
                motion_weight = self._compute_motion_weight(track, appearance_dist)

                # 融合代价
                cost = (
                    dynamic_iou_w * iou_dist +
                    self.config.mahal_weight * mahal_norm +
                    dynamic_app_w * appearance_dist +
                    self.config.social_weight * social_cost +
                    self.config.scale_weight * scale_cost +
                    motion_weight * motion_cost +
                    appearance_penalty +
                    anchor_penalty +
                    spatial_penalty
                )
                
                cost_matrix[i, j] = cost
        
        return cost_matrix
    
    def _nonlinear_penalty(
        self,
        deviation: float,
        threshold: float = 0.0,
        max_penalty: float = 1.0,
    ) -> float:
        """
        非线性惩罚函数：变化越大惩罚越大

        分段设计：
        - deviation <= threshold：无惩罚（正常范围）
        - threshold < deviation <= threshold + transition：二次增长（温和区间）
        - deviation > threshold + transition：指数增长（大偏差区间，惩罚陡增）

        这确保了中等偏差有适度惩罚，而异常大的偏差会受到显著惩罚，
        有效抑制外观相似但尺度/位置/速度严重异常的错误匹配。

        Args:
            deviation: 偏差值（已取绝对值的正数）
            threshold: 容忍阈值，低于此值无惩罚
            max_penalty: 最大惩罚值（截断上限）

        Returns:
            非线性惩罚值 [0, max_penalty]
        """
        if not self.config.nonlinear_penalty_enabled:
            # 线性回退
            if deviation <= threshold:
                return 0.0
            return min(max_penalty, deviation - threshold)

        if deviation <= threshold:
            return 0.0

        excess = deviation - threshold
        gentle = self.config.nonlinear_gentle_slope
        steep = self.config.nonlinear_steep_slope
        transition = self.config.nonlinear_transition

        if excess <= transition:
            # 二次增长：温和区间，惩罚缓慢上升
            penalty = gentle * (excess / transition) ** 2 * transition
        else:
            # 指数增长：大偏差区间，惩罚陡增
            # 先算二次区间在过渡点的值
            base_penalty = gentle * transition
            # 指数部分：从过渡点开始指数增长
            penalty = base_penalty + steep * (np.exp((excess - transition) / transition) - 1)

        return min(max_penalty, penalty)

    def _compute_scale_cost(
        self,
        track: AdvancedTrack,
        det: np.ndarray,
    ) -> float:
        """
        计算尺度一致性代价（静态检查：面积是否合理）

        使用非线性惩罚：面积比值偏离1.0越大，惩罚增长越陡峭。
        - 中等偏差（如面积比1.3~1.7）：温和惩罚
        - 较大偏差（如面积比>2.0或<0.5）：惩罚陡增

        对小目标更宽容（检测波动大），对大目标更严格。

        Args:
            track: 轨迹对象
            det: 检测框 [x1, y1, x2, y2]

        Returns:
            尺度一致性代价 [0, 1]
        """
        scale_cost = 0
        det_w = det[2] - det[0]
        det_h = det[3] - det[1]
        det_area = det_w * det_h
        if det_area > 0 and len(track.size_history) > 0:
            hist_areas = [s[0] * s[1] for s in track.size_history[-5:]]
            avg_area = np.mean(hist_areas)
            if avg_area > 0:
                area_ratio = det_area / avg_area

                # 面积比值偏离1.0的绝对偏差
                deviation = abs(area_ratio - 1.0)

                if avg_area < 500:  # 小目标（中远处）
                    # 小目标检测波动大，容忍阈值更宽
                    threshold = 0.7  # area_ratio在0.3~1.7范围内无惩罚
                elif avg_area < 2000:  # 中目标
                    threshold = 0.5  # area_ratio在0.5~1.5范围内无惩罚
                else:  # 大目标（近处）
                    threshold = 0.3  # area_ratio在0.7~1.3范围内无惩罚

                scale_cost = self._nonlinear_penalty(
                    deviation=deviation,
                    threshold=threshold,
                    max_penalty=1.0,
                )

        return scale_cost

    def _compute_motion_weight(
        self,
        track: AdvancedTrack,
        appearance_dist: float,
    ) -> float:
        """
        统一运动约束权重调整（合并2D+3D的边界处理）

        综合考虑：
        - 轨迹丢失时间：丢失越久运动信息越不可靠
        - 新轨迹：数据不足，降权
        - 外观相似度：高相似度时运动约束降权（可能是转身等合理变化）

        Args:
            track: 轨迹对象
            appearance_dist: 外观距离

        Returns:
            调整后的运动约束权重
        """
        weight = self.config.motion_consistency_weight

        # 丢失时间降权
        if track.time_since_update <= 2:
            pass  # 连续跟踪或刚丢失1-2帧，权重正常
        elif track.time_since_update <= 10:
            weight *= 0.5
        else:
            weight *= 0.2

        # 新轨迹深度约束降权（2D部分保留，3D深度部分关闭）
        if track.hits <= self.config.pseudo_3d_new_track_frames:
            # 新轨迹：只保留2D运动检查的权重（约60%），深度部分不可靠
            weight *= 0.6
        # 丢失超过阈值帧数后，深度约束额外降权
        elif track.time_since_update > self.config.pseudo_3d_lost_frames:
            weight *= 0.7  # 在丢失时间降权基础上再降

        # 外观相似度调整
        if appearance_dist < 0.1:  # similarity > 0.9
            weight *= 0.2
        elif appearance_dist < 0.2:  # similarity > 0.8
            weight *= 0.5
        elif appearance_dist > 0.5:  # similarity < 0.5
            weight *= 1.5

        return weight

    def _compute_motion_consistency_cost(
        self,
        track: AdvancedTrack,
        det: np.ndarray,
        appearance_dist: float,
    ) -> float:
        """
        统一运动一致性代价（2D方向/速度 + 伪三维深度扩展）

        无深度运动时：仅做2D方向+速度检查
        有深度运动时：升级为3D方向+速度检查，并增加深度方向矛盾检查

        速度一致性使用非线性惩罚：速度比偏离1.0越大，惩罚越陡峭。
        方向一致性使用非线性惩罚：cos_angle偏离1.0越大，惩罚越陡峭。

        与 scale_cost 的分工：
        - scale_cost：面积比值是否在合理范围（静态检查）
        - 本方法：运动方向/速度是否一致，深度方向是否矛盾（动态检查）
        两者不重复惩罚同一信息。

        Args:
            track: 轨迹对象
            det: 检测框 [x1, y1, x2, y2]
            appearance_dist: 外观距离

        Returns:
            运动一致性代价 [0, 1]
        """
        # ===== 2D运动基础检查 =====
        if not track.velocity_ema_initialized:
            return 0.0

        ema_vel = track.velocity_ema  # (vx, vy)
        ema_speed = np.linalg.norm(ema_vel)

        if ema_speed < self.config.motion_speed_threshold:
            return 0.0

        # 计算2D位移
        track_bbox = track.get_bbox()
        track_cx = (track_bbox[0] + track_bbox[2]) / 2
        track_cy = (track_bbox[1] + track_bbox[3]) / 2
        det_cx = (det[0] + det[2]) / 2
        det_cy = (det[1] + det[3]) / 2
        disp_2d = np.array([det_cx - track_cx, det_cy - track_cy], dtype=np.float32)
        det_speed_2d = np.linalg.norm(disp_2d) / max(1, track.time_since_update)

        if det_speed_2d < self.config.motion_displacement_threshold:
            return 0.0

        # 2D方向一致性（非线性惩罚：方向偏差越大惩罚越陡）
        # cos_angle: 1=同向, 0=垂直, -1=反向
        ema_dir = ema_vel / (ema_speed + 1e-8)
        disp_dir = disp_2d / (np.linalg.norm(disp_2d) + 1e-8)
        cos_angle_2d = np.dot(ema_dir, disp_dir)

        # 方向偏差：0=完美同向, 1=垂直, 2=反向
        direction_deviation_2d = 1.0 - cos_angle_2d  # [0, 2]
        # 容忍阈值：方向偏差<0.3（cos>0.7）视为正常
        direction_cost_2d = self._nonlinear_penalty(
            deviation=direction_deviation_2d,
            threshold=0.3,
            max_penalty=1.0,
        )

        # 2D速度一致性（非线性惩罚：速度比偏离1.0越大惩罚越陡）
        speed_ratio_2d = det_speed_2d / (ema_speed + 1e-8)
        # 速度比偏差：偏离1.0的量
        speed_deviation_2d = abs(speed_ratio_2d - 1.0)
        # 容忍阈值：速度比在0.5~1.5范围内（偏差<0.5）视为正常
        speed_cost_2d = self._nonlinear_penalty(
            deviation=speed_deviation_2d,
            threshold=0.5,
            max_penalty=1.0,
        )

        # 2D综合代价
        cost_2d = 0.6 * direction_cost_2d + 0.4 * speed_cost_2d

        # ===== 伪三维深度扩展 =====
        # 仅当有可靠的深度运动信息时才升级为3D检查
        depth_motion = (
            track.pseudo_3d_velocity_initialized
            and track.scale_ratio_ema_initialized
            and abs(track.scale_ratio_ema - 1.0) > self.config.pseudo_3d_scale_threshold
            and track.hits > self.config.pseudo_3d_new_track_frames
        )

        if not depth_motion:
            return min(1.0, cost_2d)

        # 有深度运动：用3D方向替代2D方向检查（避免重复）
        track_vel_3d = track.pseudo_3d_velocity  # [vx, vy, vz]
        track_speed_3d = np.linalg.norm(track_vel_3d)

        # 深度方向位移
        det_w = det[2] - det[0]
        det_h = det[3] - det[1]
        det_area = det_w * det_h
        disp_z = 0.0
        if len(track.size_history) > 0 and det_area > 1e-4:
            hist_areas = [s[0] * s[1] for s in track.size_history[-5:]]
            avg_area = np.mean(hist_areas)
            if avg_area > 1e-4:
                area_ratio = det_area / avg_area
                disp_z = (area_ratio - 1.0) * np.sqrt(avg_area)

        disp_3d = np.array([disp_2d[0], disp_2d[1], disp_z], dtype=np.float32)
        disp_3d_norm = np.linalg.norm(disp_3d)

        if disp_3d_norm < self.config.motion_displacement_threshold:
            return min(1.0, cost_2d)

        # 3D方向一致性（非线性惩罚）
        track_dir_3d = track_vel_3d / (track_speed_3d + 1e-8)
        disp_dir_3d = disp_3d / (disp_3d_norm + 1e-8)
        cos_angle_3d = np.dot(track_dir_3d, disp_dir_3d)

        direction_deviation_3d = 1.0 - cos_angle_3d
        direction_cost_3d = self._nonlinear_penalty(
            deviation=direction_deviation_3d,
            threshold=0.3,
            max_penalty=1.0,
        )

        # 3D速度一致性（非线性惩罚）
        det_speed_3d = disp_3d_norm / max(1, track.time_since_update)
        speed_ratio_3d = det_speed_3d / (track_speed_3d + 1e-8)
        speed_deviation_3d = abs(speed_ratio_3d - 1.0)
        speed_cost_3d = self._nonlinear_penalty(
            deviation=speed_deviation_3d,
            threshold=0.5,
            max_penalty=1.0,
        )

        # 深度方向矛盾检查（独立于scale_cost：scale_cost检查面积绝对值，
        # 这里检查深度方向是否与历史运动趋势矛盾）
        # 使用非线性惩罚：深度方向矛盾越大惩罚越陡
        depth_cost = 0.0
        track_vz = track_vel_3d[2]
        det_vz = disp_z / max(1, track.time_since_update)
        if track_vz * det_vz < 0:
            # 深度方向相反：用非线性惩罚，偏差越大惩罚越重
            # 方向完全相反时deviation=2（最大偏差）
            vz_conflict = abs(track_vz) + abs(det_vz)
            depth_cost = self._nonlinear_penalty(
                deviation=vz_conflict,
                threshold=0.0,
                max_penalty=1.0,
            )
        elif abs(track_vz) > 1e-4 and abs(det_vz) > 1e-4:
            vz_ratio = det_vz / (track_vz + 1e-8)
            vz_deviation = abs(vz_ratio - 1.0)
            depth_cost = self._nonlinear_penalty(
                deviation=vz_deviation,
                threshold=0.7,  # 容忍速度比在0.3~1.7范围
                max_penalty=1.0,
            )

        # 3D综合代价（用3D方向替代2D方向，避免重复）
        dir_w = self.config.pseudo_3d_direction_weight
        spd_w = 1.0 - dir_w
        depth_w = self.config.pseudo_3d_depth_cost_weight
        cost_3d = dir_w * direction_cost_3d + spd_w * speed_cost_3d + depth_w * depth_cost

        return min(1.0, cost_3d)

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
                
                # 6. 尺度一致性检查（与主代价矩阵统一逻辑）
                scale_penalty = self._compute_scale_cost(track, det)

                # 7. 伪三维深度方向惩罚
                # 遮挡恢复时，深度方向运动趋势应与历史一致
                depth_penalty = 0
                det_w = det[2] - det[0]
                det_h = det[3] - det[1]
                det_area = det_w * det_h
                if track.pseudo_3d_velocity_initialized and track.scale_ratio_ema_initialized:
                    track_vz = track.pseudo_3d_velocity[2]
                    if abs(track_vz) > 1e-2 and det_area > 0 and len(track.size_history) > 0:
                        hist_areas_r = [s[0] * s[1] for s in track.size_history[-5:]]
                        avg_area_r = np.mean(hist_areas_r)
                        if avg_area_r > 1e-4:
                            det_depth_dir = (det_area / avg_area_r - 1.0)
                            if track_vz * det_depth_dir < 0:
                                depth_penalty = 0.4
                
                # 匹配条件（满足其一即可，但运动不一致或尺度/深度不一致时更严格）
                is_match = False
                if not motion_consistent or depth_penalty > 0:
                    # 运动不一致或深度方向矛盾：要求更高的外观相似度
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
                    # 综合代价：外观 + 空间 + 运动惩罚 + 尺度惩罚 + 深度惩罚
                    cost = (1 - similarity) * 0.35 + (1 - spatial_confidence) * 0.2 + motion_penalty * 0.2 + scale_penalty * 0.15 + depth_penalty * 0.1
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
        
        # 密集场景：外观更重要
        if num_detections > self.config.density_threshold_high:
            self.adaptive_params['iou_threshold'] = 0.4
            self.adaptive_params['appearance_weight'] = 0.55
            self.adaptive_params['iou_weight'] = 0.15
        
        # 稀疏场景：IoU更可靠，但仍以外观为主
        elif num_detections < self.config.density_threshold_low:
            self.adaptive_params['iou_threshold'] = 0.2
            self.adaptive_params['appearance_weight'] = 0.4
            self.adaptive_params['iou_weight'] = 0.3
        
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