"""
跟踪参数配置模块

提供统一的参数配置接口，支持：
  1. 预设配置（快速、标准、高精度）
  2. 场景自适应配置
  3. 参数调优接口
  4. 配置文件导入导出
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import json


@dataclass
class TrackerConfig:
    """
    跟踪器配置类

    包含所有跟踪相关参数
    """

    # ==================== 基础参数 ====================
    max_age: int = 30
    """轨迹最大存活帧数（未匹配后删除）"""

    min_hits: int = 3
    """轨迹确认所需最小匹配次数"""

    iou_threshold: float = 0.3
    """IoU 匹配阈值"""

    # ==================== ByteTrack 参数 ====================
    use_bytetrack: bool = True
    """是否启用 ByteTrack 二次匹配"""

    high_conf_threshold: float = 0.5
    """高置信度检测阈值"""

    low_conf_threshold: float = 0.1
    """低置信度检测阈值（用于 ByteTrack 二次匹配）"""

    # ==================== 外观特征参数 ====================
    use_reid: bool = True
    """是否使用 ReID 外观特征"""

    reid_model: str = 'osnet_x1_0'
    """ReID 模型名称 (osnet_x1_0, osnet_x0_75, resnet50, mobilenet)"""

    appearance_weight: float = 0.4
    """外观相似度权重"""

    iou_weight: float = 0.3
    """IoU 距离权重"""

    mahal_weight: float = 0.3
    """马氏距离权重"""

    feature_smooth_alpha: float = 0.7
    """特征平滑系数（新特征权重）"""

    # ==================== 社会行为参数 ====================
    use_social_constraint: bool = True
    """是否启用社会行为约束"""

    social_weight: float = 0.2
    """社会行为代价权重"""

    overlap_threshold: float = 0.3
    """重叠惩罚阈值"""

    # ==================== 自适应参数 ====================
    use_adaptive: bool = True
    """是否启用自适应参数调整"""

    density_threshold_high: int = 20
    """高密度场景阈值（目标数）"""

    density_threshold_low: int = 5
    """低密度场景阈值（目标数）"""

    # ==================== 马氏距离门控 ====================
    gating_threshold: float = 9.4877
    """马氏距离门控阈值（卡方分布 95% 置信度）"""

    # ==================== 类别特定参数 ====================
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
    """类别特定参数"""

    # ==================== 场景参数 ====================
    scene_type: str = 'general'
    """场景类型 (general, crowded, highway, indoor, outdoor)"""

    camera_motion: bool = False
    """是否有相机运动"""

    # ==================== 输出参数 ====================
    output_trajectory: bool = False
    """是否输出轨迹历史"""

    output_confidence: bool = True
    """是否输出置信度"""


class ConfigPresets:
    """配置预设"""

    @staticmethod
    def fast() -> TrackerConfig:
        """快速模式 - 低延迟，适合实时应用"""
        return TrackerConfig(
            max_age=15,
            min_hits=2,
            iou_threshold=0.25,
            use_bytetrack=True,
            use_reid=False,  # 禁用 ReID 提升速度
            use_social_constraint=False,
            use_adaptive=False,
        )

    @staticmethod
    def standard() -> TrackerConfig:
        """标准模式 - 平衡速度和精度，优化减少ID切换"""
        return TrackerConfig(
            max_age=50,  # 增大存活期，遮挡后更易恢复
            min_hits=3,
            iou_threshold=0.3,
            use_bytetrack=True,
            use_reid=True,
            reid_model='osnet_x1_0',
            appearance_weight=0.5,  # 提高外观权重，增强ReID匹配
            iou_weight=0.3,
            mahal_weight=0.2,
            feature_smooth_alpha=0.3,  # EMA平滑系数（低值=历史特征权重70%=更稳定）
            use_social_constraint=True,
            use_adaptive=False,  # 禁用自适应，保持参数稳定
        )

    @staticmethod
    def high_precision() -> TrackerConfig:
        """高精度模式 - 最高精度，适合离线分析"""
        return TrackerConfig(
            max_age=50,
            min_hits=5,
            iou_threshold=0.35,
            use_bytetrack=True,
            use_reid=True,
            reid_model='osnet_x1_0',
            appearance_weight=0.5,
            use_social_constraint=True,
            use_adaptive=True,
            output_trajectory=True,
        )

    @staticmethod
    def crowded_scene() -> TrackerConfig:
        """拥挤场景模式"""
        return TrackerConfig(
            max_age=20,
            min_hits=2,
            iou_threshold=0.4,
            use_bytetrack=True,
            use_reid=True,
            appearance_weight=0.6,
            iou_weight=0.2,
            mahal_weight=0.2,
            use_social_constraint=True,
            social_weight=0.3,
            overlap_threshold=0.2,
            use_adaptive=True,
            density_threshold_high=30,
            density_threshold_low=10,
        )

    @staticmethod
    def highway() -> TrackerConfig:
        """高速公路场景模式"""
        return TrackerConfig(
            max_age=40,
            min_hits=3,
            iou_threshold=0.35,
            use_bytetrack=True,
            use_reid=True,
            appearance_weight=0.3,
            iou_weight=0.4,
            mahal_weight=0.3,
            use_social_constraint=False,
            use_adaptive=True,
        )

    @staticmethod
    def stable_iou() -> TrackerConfig:
        """稳定IoU模式 - 启用ReID+EMA平滑，减少ID切换"""
        return TrackerConfig(
            max_age=50,  # 增大存活期，减少遮挡丢失
            min_hits=5,  # 提高确认阈值，避免误确认
            iou_threshold=0.25,  # IoU阈值（适中）
            use_bytetrack=True,
            use_reid=True,  # 启用ReID外观特征
            reid_model='osnet_x1_0',  # 使用OSNet（行人重识别专用）
            appearance_weight=0.5,  # 外观权重（提高以稳定ID）
            iou_weight=0.3,  # IoU权重
            mahal_weight=0.2,  # 马氏距离权重（降低以减少运动干扰）
            feature_smooth_alpha=0.3,  # EMA平滑系数（低值=更稳定，历史特征权重更大）
            use_social_constraint=True,
            social_weight=0.1,
            use_adaptive=False,  # 禁用自适应，保持稳定
        )


class ConfigManager:
    """
    配置管理器

    提供配置的创建、加载、保存、调优等功能
    """

    PRESETS = {
        'fast': ConfigPresets.fast,
        'standard': ConfigPresets.standard,
        'high_precision': ConfigPresets.high_precision,
        'crowded_scene': ConfigPresets.crowded_scene,
        'highway': ConfigPresets.highway,
        'stable_iou': ConfigPresets.stable_iou,  # 新增：稳定IoU模式
    }

    def __init__(self, config: Optional[TrackerConfig] = None):
        """
        初始化配置管理器

        Args:
            config: 初始配置，默认使用标准配置
        """
        self.config = config or ConfigPresets.standard()
        self.config_history: List[TrackerConfig] = []

    def get_preset(self, preset_name: str) -> TrackerConfig:
        """
        获取预设配置

        Args:
            preset_name: 预设名称

        Returns:
            配置对象
        """
        if preset_name not in self.PRESETS:
            raise ValueError(f"未知预设: {preset_name}，可用预设: {list(self.PRESETS.keys())}")

        return self.PRESETS[preset_name]()

    def set_preset(self, preset_name: str):
        """
        设置预设配置

        Args:
            preset_name: 预设名称
        """
        self._save_history()
        self.config = self.get_preset(preset_name)

    def update(self, **kwargs):
        """
        更新配置参数

        Args:
            **kwargs: 参数键值对
        """
        self._save_history()

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"警告: 未知参数 {key}")

    def _save_history(self):
        """保存配置历史"""
        self.config_history.append(TrackerConfig(**asdict(self.config)))
        if len(self.config_history) > 10:
            self.config_history = self.config_history[-10:]

    def rollback(self):
        """回滚到上一个配置"""
        if self.config_history:
            self.config = self.config_history.pop()
        else:
            print("没有可回滚的配置")

    def save(self, path: str):
        """
        保存配置到文件

        Args:
            path: 文件路径 (支持 .yaml 和 .json)
        """
        path = Path(path)
        config_dict = asdict(self.config)

        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        elif path.suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

        print(f"配置已保存到: {path}")

    def load(self, path: str):
        """
        从文件加载配置

        Args:
            path: 文件路径
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        self._save_history()

        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")

        self.config = TrackerConfig(**config_dict)
        print(f"配置已加载: {path}")

    def adapt_to_scene(self, num_detections: int, avg_motion: float = 0):
        """
        根据场景自适应调整参数

        Args:
            num_detections: 当前检测数量
            avg_motion: 平均运动幅度
        """
        if not self.config.use_adaptive:
            return

        self._save_history()

        # 根据密度调整
        if num_detections > self.config.density_threshold_high:
            # 密集场景
            self.config.iou_threshold = min(0.5, self.config.iou_threshold + 0.1)
            self.config.appearance_weight = min(0.6, self.config.appearance_weight + 0.1)
        elif num_detections < self.config.density_threshold_low:
            # 稀疏场景
            self.config.iou_threshold = max(0.2, self.config.iou_threshold - 0.1)
            self.config.appearance_weight = max(0.2, self.config.appearance_weight - 0.1)

        # 根据运动幅度调整
        if avg_motion > 10:  # 快速运动
            self.config.max_age = max(15, self.config.max_age - 5)
        elif avg_motion < 3:  # 缓慢运动
            self.config.max_age = min(50, self.config.max_age + 5)

    def get_class_config(self, class_name: str) -> Dict[str, Any]:
        """
        获取类别特定配置

        Args:
            class_name: 类别名称

        Returns:
            配置字典
        """
        if class_name in self.config.class_specific_params:
            return self.config.class_specific_params[class_name]
        return {}

    def tune_for_metric(self, metric_name: str, target_value: float):
        """
        针对特定指标调优参数

        Args:
            metric_name: 指标名称 (mota, motp, idf1, ids)
            target_value: 目标值
        """
        self._save_history()

        if metric_name == 'mota':
            # 提高 MOTA：减少漏检
            self.config.iou_threshold = max(0.2, self.config.iou_threshold - 0.05)
            self.config.max_age = min(50, self.config.max_age + 5)
            self.config.low_conf_threshold = max(0.05, self.config.low_conf_threshold - 0.02)

        elif metric_name == 'motp':
            # 提高 MOTP：提高定位精度
            self.config.iou_threshold = min(0.4, self.config.iou_threshold + 0.05)
            self.config.min_hits = max(2, self.config.min_hits - 1)

        elif metric_name == 'idf1':
            # 提高 IDF1：减少 ID 切换
            self.config.appearance_weight = min(0.6, self.config.appearance_weight + 0.1)
            self.config.use_reid = True
            self.config.min_hits = min(5, self.config.min_hits + 1)

        elif metric_name == 'ids':
            # 减少 ID 切换
            self.config.appearance_weight = min(0.7, self.config.appearance_weight + 0.15)
            self.config.use_reid = True
            self.config.feature_smooth_alpha = min(0.9, self.config.feature_smooth_alpha + 0.1)

    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return asdict(self.config)

    def print_config(self):
        """打印当前配置"""
        print("=" * 50)
        print("当前跟踪器配置")
        print("=" * 50)
        for key, value in asdict(self.config).items():
            if key != 'class_specific_params':
                print(f"  {key}: {value}")
        print("=" * 50)


def create_config_from_dict(config_dict: Dict[str, Any]) -> TrackerConfig:
    """
    从字典创建配置

    Args:
        config_dict: 配置字典

    Returns:
        配置对象
    """
    return TrackerConfig(**config_dict)


# 便捷函数
def get_config(preset: str = 'standard') -> TrackerConfig:
    """
    获取配置

    Args:
        preset: 预设名称

    Returns:
        配置对象
    """
    return ConfigManager().get_preset(preset)