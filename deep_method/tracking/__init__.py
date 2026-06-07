"""
深度学习跟踪模块

提供跟踪算法：
  - AdvancedTracker: 增强版跟踪器（ByteTrack + ReID + 社会约束）

使用方法：
  from deep_method.tracking import AdvancedTracker, ReIDExtractor, get_config
  
  # 获取配置
  config = get_config('standard')
  
  # 创建跟踪器
  tracker = AdvancedTracker(config)
  
  # 设置 ReID 提取器
  reid = ReIDExtractor(model_name='osnet_x1_0')
  tracker.set_reid_extractor(reid)
  
  # 更新跟踪
  tracks = tracker.update(detections, confidences, features, image)
"""

from deep_method.tracking.kalman_filter import KalmanFilter

from deep_method.tracking.advanced_tracker import (
    AdvancedTracker,
    AdvancedTrackerConfig,
    AdvancedTrack,
    TrackState,
)

from deep_method.tracking.reid_extractor import (
    ReIDExtractor,
    FeatureBuffer,
)

from deep_method.tracking.tracking_config import (
    TrackerConfig,
    ConfigPresets,
    ConfigManager,
    get_config,
    create_config_from_dict,
)


# 模块版本
__version__ = '1.0.0'

# 导出的类和函数
__all__ = [
    # 卡尔曼滤波器
    'KalmanFilter',
    
    # 跟踪器
    'AdvancedTracker',
    'AdvancedTrackerConfig',
    'AdvancedTrack',
    'TrackState',
    
    # ReID
    'ReIDExtractor',
    'FeatureBuffer',
    
    # 配置
    'TrackerConfig',
    'ConfigPresets',
    'ConfigManager',
    'get_config',
    'create_config_from_dict',
]


def create_tracker(
    tracker_type: str = 'advanced',
    preset: str = 'standard',
    use_reid: bool = True,
    reid_model: str = 'osnet_x1_0',
    device: str = 'cuda:0',
    use_torchreid: bool = True,
) -> AdvancedTracker:
    """
    快速创建跟踪器
    
    Args:
        tracker_type: 跟踪器类型 (仅支持 'advanced')
        preset: 配置预设 (fast, standard, high_precision, crowded_scene, highway, stable_iou)
        use_reid: 是否使用 ReID
        reid_model: ReID 模型名称 (osnet_x1_0推荐，有MSMT17预训练权重)
        device: 设备
        use_torchreid: 是否使用torchreid库（True=使用OSNet，False=使用torchvision ResNet50）
    
    Returns:
        跟踪器实例
    """
    tracker_config = get_config(preset)
    
    # 转换为 AdvancedTrackerConfig
    advanced_config = AdvancedTrackerConfig(
        max_age=tracker_config.max_age,
        min_hits=tracker_config.min_hits,
        iou_threshold=tracker_config.iou_threshold,
        use_bytetrack=tracker_config.use_bytetrack,
        high_conf_threshold=tracker_config.high_conf_threshold,
        low_conf_threshold=tracker_config.low_conf_threshold,
        use_reid=use_reid,
        reid_model=reid_model,
        appearance_weight=tracker_config.appearance_weight,
        iou_weight=tracker_config.iou_weight,
        mahal_weight=tracker_config.mahal_weight,
        feature_smooth_alpha=tracker_config.feature_smooth_alpha,
        use_social_constraint=tracker_config.use_social_constraint,
        social_weight=tracker_config.social_weight,
        overlap_threshold=tracker_config.overlap_threshold,
        use_adaptive=tracker_config.use_adaptive,
        density_threshold_high=tracker_config.density_threshold_high,
        density_threshold_low=tracker_config.density_threshold_low,
        gating_threshold=tracker_config.gating_threshold,
        scene_type=tracker_config.scene_type,
        camera_motion=tracker_config.camera_motion,
        output_trajectory=tracker_config.output_trajectory,
        output_confidence=tracker_config.output_confidence,
        class_specific_params=tracker_config.class_specific_params,
    )
    
    tracker = AdvancedTracker(advanced_config)
    if use_reid:
        reid = ReIDExtractor(
            model_name=reid_model, 
            device=device,
            use_torchreid=use_torchreid,
            smooth_alpha=tracker_config.feature_smooth_alpha,
        )
        tracker.set_reid_extractor(reid)
    return tracker


def get_available_presets() -> list:
    """获取可用的配置预设"""
    return ['fast', 'standard', 'high_precision', 'crowded_scene', 'highway']


def get_available_reid_models() -> list:
    """获取可用的 ReID 模型"""
    return ['osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'resnet50', 'mobilenet']