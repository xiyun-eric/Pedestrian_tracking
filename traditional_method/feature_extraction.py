"""
传统CV特征提取与 ReID 模块

核心方法:
  1. 颜色直方图 (Color Histogram): 描述目标颜色分布
     - HSV 空间分别统计 H(30bin) + S(32bin) + V(32bin)
     - 对光照变化有一定鲁棒性
  
  2. HOG 特征 (Histogram of Oriented Gradients): 描述目标形状轮廓
     - 支持手动实现和OpenCV API两种模式
     - 对颜色变化不敏感
  
  3. 融合特征: 颜色 + 形状 加权组合
     - 综合外观和几何信息
     - 提升 ReID 准确性

ReID 匹配策略:
  - 特征余弦相似度 (Cosine Similarity)
  - 特征欧氏距离 (Euclidean Distance)
  - 自适应阈值匹配
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# 导入HOG特征提取器
from traditional_method.hog_detector import HOGFeatureExtractor


@dataclass
class FeatureConfig:
    """特征提取配置"""
    use_color_hist: bool = True
    use_hog: bool = True
    hist_h_bins: int = 30      # H 通道 bins
    hist_s_bins: int = 32      # S 通道 bins
    hist_v_bins: int = 32      # V 通道 bins
    hog_win_size: Tuple[int, int] = (64, 128)
    color_weight: float = 0.4   # 颜色特征权重
    hog_weight: float = 0.6     # HOG 特征权重
    match_threshold: float = 0.6  # 匹配相似度阈值


class FeatureExtractor:
    """
    传统CV特征提取器
    
    提取:
    - HSV 颜色直方图 (94维)
    - HOG 形状特征 (3780维) - 支持手动实现和OpenCV API两种模式
    - 融合特征向量 (归一化后)
    """
    
    def __init__(self, config: FeatureConfig = None, use_opencv_api: bool = True):
        self.config = config or FeatureConfig()
        self.use_opencv_api = use_opencv_api
        
        if self.config.use_hog:
            # 使用HOG特征提取器（支持手动实现和OpenCV API两种模式）
            self.hog_extractor = HOGFeatureExtractor(
                win_size=self.config.hog_win_size,
                cell_size=(8, 8),
                block_size=(2, 2),
                block_stride=(1, 1),
                nbins=9,
                use_opencv_api=use_opencv_api,
            )
    
    def extract_color_histogram(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        提取 HSV 颜色直方图特征
        
        原理:
        - 将图像转到 HSV 色彩空间（更符合人眼感知）
        - 分别对 H、S、V 三个通道统计直方图
        - 归一化后拼接为特征向量
        
        Args:
            image: BGR 图像
            bbox: [x1, y1, x2, y2]
        
        Returns:
            (94,) 颜色直方图特征向量 (30+32+32)
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(
                self.config.hist_h_bins + self.config.hist_s_bins + self.config.hist_v_bins,
                dtype=np.float32
            )
        
        patch = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        
        # 分别计算三个通道的直方图
        h_hist = cv2.calcHist([hsv], [0], None, [self.config.hist_h_bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [self.config.hist_s_bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [self.config.hist_v_bins], [0, 256])
        
        # 归一化
        cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)
        
        feature = np.concatenate([
            h_hist.flatten(),
            s_hist.flatten(),
            v_hist.flatten(),
        ]).astype(np.float32)
        
        return feature
    
    def extract_hog_features(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        提取 HOG 特征
        
        原理:
        - 裁剪目标区域，缩放到固定尺寸 (64x128)
        - 计算梯度方向直方图
        - HOG 描述符编码了目标的形状和轮廓信息
        
        Args:
            image: BGR 图像
            bbox: [x1, y1, x2, y2]
        
        Returns:
            (3780,) HOG 特征向量
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(3780, dtype=np.float32)
        
        patch = image[y1:y2, x1:x2]
        patch = cv2.resize(patch, self.config.hog_win_size)
        
        # 使用HOG特征提取器（根据use_opencv_api决定使用API还是手动实现）
        features = self.hog_extractor.compute(patch)
        
        # L2 归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.astype(np.float32)
    
    def extract_features(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        提取融合特征向量
        
        Args:
            image: BGR 图像
            bbox: [x1, y1, x2, y2]
        
        Returns:
            融合后的归一化特征向量
        """
        features = []
        weights = []
        
        if self.config.use_color_hist:
            color_feat = self.extract_color_histogram(image, bbox)
            features.append(color_feat * self.config.color_weight)
            weights.append(self.config.color_weight)
        
        if self.config.use_hog:
            hog_feat = self.extract_hog_features(image, bbox)
            features.append(hog_feat * self.config.hog_weight)
            weights.append(self.config.hog_weight)
        
        if not features:
            return np.array([], dtype=np.float32)
        
        combined = np.concatenate(features)
        
        # 整体归一化
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined


class ReIDMatcher:
    """
    重识别匹配器
    
    用于遮挡后目标重新出现的身份关联。
    
    匹配策略（级联）:
    1. 余弦相似度 >= high_threshold -> 直接匹配
    2. 余弦相似度 >= low_threshold -> 候选匹配
    3. 结合空间距离二次筛选
    """
    
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        high_threshold: float = 0.85,
        low_threshold: float = 0.5,
        max_history: int = 30,
    ):
        self.extractor = feature_extractor
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_history = max_history
        
        # 存储每个 track_id 的历史特征库
        self._feature_gallery: Dict[int, List[np.ndarray]] = {}
    
    def register(self, track_id: int, features: np.ndarray):
        """注册/更新目标的特征"""
        if track_id not in self._feature_gallery:
            self._feature_gallery[track_id] = []
        
        self._feature_gallery[track_id].append(features)
        
        # 限制历史特征数量
        if len(self._feature_gallery[track_id]) > self.max_history:
            self._feature_gallery[track_id] = self._feature_gallery[track_id][-self.max_history:]
    
    def match(
        self, 
        query_features: np.ndarray,
        candidate_ids: List[int],
    ) -> Optional[int]:
        """
        在候选ID中寻找最匹配的目标
        
        Args:
            query_features: 查询目标的特征
            candidate_ids: 候选 track_id 列表
        
        Returns:
            最匹配的 track_id，无匹配返回 None
        """
        if not candidate_ids or len(query_features) == 0:
            return None
        
        best_id = None
        best_sim = -1.0
        
        for tid in candidate_ids:
            if tid not in self._feature_gallery:
                continue
            
            # 与历史特征库中每个特征计算相似度，取最大
            gallery_features = self._feature_gallery[tid]
            
            for gf in gallery_features[-5:]:  # 只用最近5个特征
                sim = self._cosine_similarity(query_features, gf)
                if sim > best_sim:
                    best_sim = sim
                    best_id = tid
        
        if best_sim >= self.high_threshold:
            return best_id
        elif best_sim >= self.low_threshold:
            return best_id  # 低阈值也返回，但标记为不确定
        else:
            return None
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        if len(a) == 0 or len(b) == 0:
            return 0.0
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
    
    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """计算欧氏距离"""
        return float(np.linalg.norm(a - b))
    
    def remove(self, track_id: int):
        """移除目标的特征记录"""
        self._feature_gallery.pop(track_id, None)
    
    def clear(self):
        """清空所有特征记录"""
        self._feature_gallery.clear()
    
    @property
    def registered_count(self) -> int:
        return len(self._feature_gallery)
