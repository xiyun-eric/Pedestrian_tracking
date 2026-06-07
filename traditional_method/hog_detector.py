"""
HOG + SVM 行人检测器（传统CV方法）

核心原理:
  - HOG (Histogram of Oriented Gradients): 方向梯度直方图特征
    通过统计局部区域梯度方向来刻画目标的外观形状
    对光照变化和几何形变具有较好的不变性
  
  - SVM (Support Vector Machine): 支持向量机分类器
    寻找最优分类超平面，最大化正负样本间隔

实现细节:
  - 使用 OpenCV 内置的 HOGDescriptor + 预训练行人检测器
  - 支持多尺度检测 (image pyramid)
  - NMS 后处理去除冗余框
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class HOGDetector:
    """
    HOG + SVM 行人检测器
    
    使用 OpenCV 预训练的 Dalal-Triggs 行人检测器。
    该检测器在 INRIA Person Dataset 上训练，是最经典的行人检测方法之一。
    
    算法流程:
    1. 计算图像 HOG 特征（9个方向 bins, 8x8 cell, 2x2 block）
    2. 用滑动窗口扫描特征图
    3. SVM 分类器判断每个窗口是否包含行人
    4. NMS 去重
    """
    
    def __init__(
        self,
        win_stride: Tuple[int, int] = (8, 8),
        padding: Tuple[int, int] = (16, 16),
        scale: float = 1.05,
        hit_threshold: float = 0.0,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.3,
    ):
        """
        Args:
            win_stride: 滑动窗口步长 (x, y)，越小检测越密但越慢
            padding: 边缘填充，帮助检测图像边缘的目标
            scale: 图像金字塔缩放系数，< 1.05 更精细但更慢
            hit_threshold: SVM 得分阈值 (OpenCV 内部使用)
            conf_threshold: 最终置信度阈值 (0~1)
            nms_threshold: NMS IoU 阈值
        """
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.hit_threshold = hit_threshold
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
    
    def detect(
        self, 
        image: np.ndarray, 
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行行人检测
        
        Args:
            image: BGR 图像 (H, W, 3)
            roi: 感兴趣区域 (x, y, w, h)，None 则全图检测
        
        Returns:
            detections: (N, 4) 边界框数组 [x1, y1, x2, y2]
            confidences: (N,) 置信度数组
        """
        if roi is not None:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        # HOG 多尺度检测
        # detectMultiScale 内部实现图像金字塔 + 滑动窗口
        rects, weights = self.hog.detectMultiScale(
            image,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
            hitThreshold=self.hit_threshold,
        )
        
        if len(rects) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        
        # OpenCV 返回 (x, y, w, h)，转换为 [x1, y1, x2, y2]
        detections = np.array([
            [x, y, x + w, y + h] for (x, y, w, h) in rects
        ], dtype=np.float32)
        
        # 归一化置信度到 [0, 1]
        if len(weights) > 0:
            confidences = np.array(weights, dtype=np.float32)
            # HOG的weights不是概率，做简单归一化
            confidences = np.clip(confidences / (np.max(confidences) + 1e-8), 0, 1)
        else:
            confidences = np.ones(len(detections), dtype=np.float32)
        
        # 按置信度过滤
        mask = confidences >= self.conf_threshold
        detections = detections[mask]
        confidences = confidences[mask]
        
        # NMS 去重
        if len(detections) > 1:
            keep = self._nms(detections, confidences, self.nms_threshold)
            detections = detections[keep]
            confidences = confidences[keep]
        
        # 如果使用了 ROI，修正坐标
        if roi is not None:
            detections[:, 0] += roi[0]
            detections[:, 1] += roi[1]
            detections[:, 2] += roi[0]
            detections[:, 3] += roi[1]
        
        return detections, confidences
    
    @staticmethod
    def _nms(
        boxes: np.ndarray, 
        scores: np.ndarray, 
        threshold: float
    ) -> np.ndarray:
        """
        非极大值抑制 (Non-Maximum Suppression)
        
        原理:
        1. 按得分降序排列
        2. 得分最高的框保留
        3. 移除与该框 IoU > threshold 的其他框
        4. 重复直到所有框处理完
        
        Args:
            boxes: (N, 4) [x1, y1, x2, y2]
            scores: (N,) 得分
            threshold: IoU 阈值
        
        Returns:
            保留的框索引
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # 计算当前框与剩余框的 IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            keep_idx = np.where(iou <= threshold)[0]
            order = order[keep_idx + 1]
        
        return np.array(keep, dtype=np.int32)
    
    def get_hog_features(
        self, 
        image: np.ndarray, 
        bbox: np.ndarray
    ) -> np.ndarray:
        """
        提取指定区域的 HOG 特征向量（用于 ReID）
        
        Args:
            image: BGR 图像
            bbox: [x1, y1, x2, y2]
        
        Returns:
            HOG 特征向量
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(3780, dtype=np.float32)
        
        patch = image[y1:y2, x1:x2]
        # 缩放到固定大小
        patch = cv2.resize(patch, (64, 128))
        
        # 计算 HOG 描述符
        hog_descriptor = cv2.HOGDescriptor(
            _winSize=(64, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        features = hog_descriptor.compute(patch)
        return features.flatten()


def test_hog_detector():
    """测试 HOG 检测器"""
    import sys
    from pathlib import Path
    
    # 找一张测试图
    test_dir = Path(__file__).resolve().parents[1] / "data" / "kitti" / "images"
    seq_dirs = sorted(test_dir.glob("*")) if test_dir.exists() else []
    
    if not seq_dirs:
        print("未找到 KITTI 图像，请先下载数据集")
        return
    
    img_path = sorted(seq_dirs[0].glob("*.png"))[0]
    image = cv2.imread(str(img_path))
    
    if image is None:
        print(f"无法读取图像: {img_path}")
        return
    
    detector = HOGDetector()
    detections, confidences = detector.detect(image)
    
    print(f"检测到 {len(detections)} 个行人")
    for d, c in zip(detections[:5], confidences[:5]):
        print(f"  框: {d}, 置信度: {c:.3f}")


if __name__ == "__main__":
    test_hog_detector()
