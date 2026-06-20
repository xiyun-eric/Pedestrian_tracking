"""
HOG + SVM 行人检测器（支持手动实现和OpenCV API两种模式）

核心原理:
  - HOG (Histogram of Oriented Gradients): 方向梯度直方图特征
    通过统计局部区域梯度方向来刻画目标的外观形状
    对光照变化和几何形变具有较好的不变性
  
  - SVM (Support Vector Machine): 支持向量机分类器
    寻找最优分类超平面，最大化正负样本间隔

实现模式:
  - use_opencv_api=True (默认): 使用OpenCV的HOGDescriptor和SVM，速度快
  - use_opencv_api=False: 手动实现HOG特征提取和SVM决策函数，可验证算法原理

手动实现细节:
  - HOG特征提取: 使用numpy向量化操作实现梯度计算、Cell直方图、Block归一化
  - SVM分类: 手动实现决策函数（使用预训练权重）
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class HOGFeatureExtractor:
    """
    HOG特征提取器（支持手动实现和OpenCV API两种模式）
    """
    
    def __init__(
        self,
        win_size: Tuple[int, int] = (64, 128),
        cell_size: Tuple[int, int] = (8, 8),
        block_size: Tuple[int, int] = (2, 2),
        block_stride: Tuple[int, int] = (1, 1),
        nbins: int = 9,
        use_opencv_api: bool = True,
    ):
        """
        Args:
            win_size: 检测窗口大小 (width, height)
            cell_size: Cell大小 (width, height)
            block_size: Block大小，单位为cells (width, height)
            block_stride: Block步长，单位为cells (width, height)
            nbins: 方向直方图bins数量
            use_opencv_api: 是否使用OpenCV API（默认True，速度快）
        """
        self.win_size = win_size
        self.cell_size = cell_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.nbins = nbins
        self.use_opencv_api = use_opencv_api
        
        # 计算窗口内的cell数量
        self.cells_x = win_size[0] // cell_size[0]
        self.cells_y = win_size[1] // cell_size[1]
        
        # 计算Block数量
        self.blocks_x = (self.cells_x - block_size[0]) // block_stride[0] + 1
        self.blocks_y = (self.cells_y - block_size[1]) // block_stride[1] + 1
        
        # 特征维度
        self.feature_dim = self.blocks_x * self.blocks_y * block_size[0] * block_size[1] * nbins
        
        if use_opencv_api:
            # 使用OpenCV的HOGDescriptor
            self.hog = cv2.HOGDescriptor(
                _winSize=win_size,
                _blockSize=(block_size[0] * cell_size[0], block_size[1] * cell_size[1]),
                _blockStride=(block_stride[0] * cell_size[0], block_stride[1] * cell_size[1]),
                _cellSize=cell_size,
                _nbins=nbins,
            )
    
    def compute(self, image: np.ndarray) -> np.ndarray:
        """
        计算图像的 HOG 特征向量
        
        Args:
            image: 灰度图像 (H, W) 或 BGR 图像 (H, W, 3)
        
        Returns:
            (feature_dim,) HOG 特征向量
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 确保图像是uint8类型（OpenCV HOGDescriptor要求）
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        # 缩放到窗口大小
        if gray.shape[1] != self.win_size[0] or gray.shape[0] != self.win_size[1]:
            gray = cv2.resize(gray, self.win_size)
        
        if self.use_opencv_api:
            # 使用OpenCV API
            features = self.hog.compute(gray)
            return features.flatten().astype(np.float32)
        else:
            # 手动实现（需要float32）
            gray_float = gray.astype(np.float32)
            return self._compute_hog_manual(gray_float)
    
    def _bgr2gray(self, image: np.ndarray) -> np.ndarray:
        """BGR转灰度"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def _compute_hog_manual(self, gray: np.ndarray) -> np.ndarray:
        """
        手动实现HOG特征提取（使用numpy向量化操作）
        """
        # 1. 计算梯度（使用OpenCV的Sobel，但后续处理手动实现）
        # 注意：完全手动实现梯度计算太慢，这里使用OpenCV的Sobel
        # 但直方图统计和归一化是手动实现的
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
        
        # 2. 计算梯度幅值和方向
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi  # 转换为角度
        angle = angle % 180  # 映射到 [0, 180)
        
        # 3. 计算每个Cell的直方图（向量化实现）
        cell_hists = self._compute_cell_histograms_vectorized(magnitude, angle)
        
        # 4. Block归一化
        features = self._normalize_blocks(cell_hists)
        
        return features
    
    def _compute_cell_histograms_vectorized(
        self, 
        magnitude: np.ndarray, 
        angle: np.ndarray
    ) -> np.ndarray:
        """
        向量化计算每个Cell的梯度方向直方图
        """
        h, w = magnitude.shape
        cell_w, cell_h = self.cell_size
        
        cell_hists = np.zeros((self.cells_y, self.cells_x, self.nbins), dtype=np.float32)
        
        bin_width = 180.0 / self.nbins
        
        for cy in range(self.cells_y):
            for cx in range(self.cells_x):
                # Cell区域
                y1, y2 = cy * cell_h, (cy + 1) * cell_h
                x1, x2 = cx * cell_w, (cx + 1) * cell_w
                
                cell_mag = magnitude[y1:y2, x1:x2].flatten()
                cell_ang = angle[y1:y2, x1:x2].flatten()
                
                # 向量化双线性插值投票
                bin_idx = cell_ang / bin_width
                bin0 = np.floor(bin_idx).astype(np.int32) % self.nbins
                bin1 = (bin0 + 1) % self.nbins
                
                w0 = 1.0 - (bin_idx - bin0)
                w1 = 1.0 - w0
                
                # 使用np.add.at进行直方图累加
                np.add.at(cell_hists[cy, cx], bin0, cell_mag * w0)
                np.add.at(cell_hists[cy, cx], bin1, cell_mag * w1)
        
        return cell_hists
    
    def _normalize_blocks(self, cell_hists: np.ndarray) -> np.ndarray:
        """
        Block归一化 (L2-Hys)
        """
        features = []
        
        for by in range(self.blocks_y):
            for bx in range(self.blocks_x):
                y1 = by * self.block_stride[1]
                x1 = bx * self.block_stride[0]
                y2 = y1 + self.block_size[1]
                x2 = x1 + self.block_size[0]
                
                block = cell_hists[y1:y2, x1:x2, :].flatten()
                
                # L2归一化
                norm = np.sqrt(np.sum(block**2) + 1e-6)
                block = block / norm
                
                # L2-Hys: 截断到0.2后重新归一化
                block = np.clip(block, 0, 0.2)
                norm = np.sqrt(np.sum(block**2) + 1e-6)
                block = block / norm
                
                features.append(block)
        
        return np.concatenate(features).astype(np.float32)


class LinearSVMClassifier:
    """
    线性SVM分类器（支持手动实现和OpenCV API两种模式）
    """
    
    def __init__(
        self,
        weights: np.ndarray,
        bias: float = 0.0,
        use_svm_api: bool = True,
    ):
        """
        Args:
            weights: 权重向量 (feature_dim,)
            bias: 偏置项
            use_svm_api: 是否使用OpenCV SVM API（默认True）
                        注意：无论True或False，决策函数都是手动实现的点积
                        此参数主要用于标识是否使用OpenCV预训练权重
        """
        self.weights = weights
        self.bias = bias
        self.use_svm_api = use_svm_api
    
    def predict(self, features: np.ndarray) -> float:
        """
        计算SVM决策函数值
        
        decision = w · x + b
        
        Args:
            features: 特征向量
        
        Returns:
            决策函数值，>0 表示正类
        """
        # 手动实现决策函数（不依赖任何SVM API）
        return np.dot(features, self.weights) + self.bias
    
    def predict_proba(self, features: np.ndarray) -> float:
        """
        计算分类概率（使用sigmoid）
        """
        decision = self.predict(features)
        return 1.0 / (1.0 + np.exp(-decision))


class HOGDetector:
    """
    HOG + SVM 行人检测器（支持手动实现和OpenCV API两种模式）
    
    默认使用OpenCV API（速度快），可分别设置HOG和SVM是否使用API。
    """
    
    # 行人典型宽高比范围
    ASPECT_RATIO_MIN = 0.2
    ASPECT_RATIO_MAX = 0.85
    
    def __init__(
        self,
        win_stride: Tuple[int, int] = (8, 8),
        padding: Tuple[int, int] = (16, 16),
        scale: float = 1.05,
        hit_threshold: float = 0.0,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.45,
        group_threshold: int = 2,
        group_eps: float = 0.5,
        aspect_ratio_min: float = 0.2,
        aspect_ratio_max: float = 0.85,
        split_merged_boxes: bool = True,
        use_hog_api: bool = True,   # HOG是否使用OpenCV API
        use_svm_api: bool = True,   # SVM是否使用OpenCV预训练权重
    ):
        """
        Args:
            win_stride: 滑动窗口步长 (x, y)
            padding: 边缘填充
            scale: 图像金字塔缩放系数
            hit_threshold: SVM 得分阈值
            conf_threshold: 最终置信度阈值 (0~1)
            nms_threshold: NMS IoU 阈值
            group_threshold: 检测框分组阈值
            group_eps: 分组相对距离阈值
            aspect_ratio_min: 行人最小宽高比
            aspect_ratio_max: 行人最大宽高比
            split_merged_boxes: 是否拆分融合框
            use_hog_api: HOG特征提取是否使用OpenCV API（默认True，速度快）
            use_svm_api: SVM是否使用OpenCV预训练权重（默认True）
        """
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.hit_threshold = hit_threshold
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.group_threshold = group_threshold
        self.group_eps = group_eps
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max
        self.split_merged_boxes = split_merged_boxes
        self.use_hog_api = use_hog_api
        self.use_svm_api = use_svm_api
        
        # 加载预训练权重（无论是否使用API都需要）
        detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
        self.svm_weights = detector[:-1].astype(np.float32)
        self.svm_bias = float(detector[-1])
        
        if use_hog_api:
            # 使用OpenCV的HOGDescriptor
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        else:
            # 使用手动实现的HOG特征提取器
            self.hog_extractor = HOGFeatureExtractor(
                win_size=(64, 128),
                cell_size=(8, 8),
                block_size=(2, 2),
                block_stride=(1, 1),
                nbins=9,
                use_opencv_api=False,
            )
        
        # SVM分类器（决策函数始终是手动实现的点积）
        self.svm = LinearSVMClassifier(
            weights=self.svm_weights,
            bias=self.svm_bias,
            use_svm_api=use_svm_api,
        )
    
    def detect(
        self, 
        image: np.ndarray, 
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行行人检测
        
        Args:
            image: BGR 图像 (H, W, 3)
            roi: 感兴趣区域 (x, y, w, h)
        
        Returns:
            detections: (N, 4) 边界框 [x1, y1, x2, y2]
            confidences: (N,) 置信度
        """
        if roi is not None:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        if self.use_hog_api:
            # 使用OpenCV API
            rects, weights = self.hog.detectMultiScale(
                image,
                winStride=self.win_stride,
                padding=self.padding,
                scale=self.scale,
                hitThreshold=self.hit_threshold,
            )
            
            if len(rects) == 0:
                return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
            
            # groupRectangles 预合并
            if self.group_threshold > 0 and len(rects) > 1:
                rects_i32 = rects.astype(np.int32)
                rects_grouped, weights_grouped = cv2.groupRectangles(
                    rects_i32, self.group_threshold, self.group_eps
                )
                if len(rects_grouped) > 0:
                    rects = rects_grouped.astype(np.float32)
                    weights = weights_grouped
            
            detections = np.array([
                [x, y, x + w, y + h] for (x, y, w, h) in rects
            ], dtype=np.float32)
            
            # Sigmoid置信度归一化
            if len(weights) > 0:
                confidences = np.array(weights, dtype=np.float32).flatten()
                confidences = 1.0 / (1.0 + np.exp(-confidences))
            else:
                confidences = np.ones(len(detections), dtype=np.float32)
        else:
            # 使用手动实现
            detections, confidences = self._detect_manual(image)
        
        # 置信度过滤
        mask = confidences >= self.conf_threshold
        detections = detections[mask]
        confidences = confidences[mask]
        
        # 宽高比过滤
        detections, confidences = self._filter_by_aspect_ratio(detections, confidences)
        
        # NMS去重
        if len(detections) > 1:
            keep = self._nms(detections, confidences, self.nms_threshold)
            detections = detections[keep]
            confidences = confidences[keep]
        
        # 融合框拆分
        if self.split_merged_boxes and len(detections) > 0:
            detections, confidences = self._split_merged_boxes(detections, confidences)
        
        # 修正ROI偏移
        if roi is not None:
            detections[:, 0] += roi[0]
            detections[:, 1] += roi[1]
            detections[:, 2] += roi[0]
            detections[:, 3] += roi[1]
        
        return detections, confidences
    
    def _detect_manual(
        self, 
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        手动实现多尺度滑动窗口检测
        
        警告：此方法非常慢，仅用于验证算法原理。
        对于1280x720图像，约需处理14400+个窗口。
        建议减少处理帧数（--frames 10）。
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        detections = []
        confidences = []
        
        # 图像金字塔
        scale_factor = 1.0
        scaled_img = gray.copy()
        
        # 限制金字塔层数以加快速度
        max_levels = 5
        level = 0
        
        while min(scaled_img.shape) >= 128 and level < max_levels:
            # 滑动窗口检测
            h, w = scaled_img.shape
            
            # 增大步长以加快速度（手动模式下使用16而不是8）
            stride = max(self.win_stride[0], 16)
            
            for y in range(0, h - 128, stride):
                for x in range(0, w - 64, stride):
                    window = scaled_img[y:y+128, x:x+64]
                    
                    # 计算HOG特征
                    features = self.hog_extractor.compute(window)
                    
                    # SVM分类
                    score = self.svm.predict(features)
                    
                    if score > 0:
                        x1 = int(x * scale_factor)
                        y1 = int(y * scale_factor)
                        x2 = int((x + 64) * scale_factor)
                        y2 = int((y + 128) * scale_factor)
                        
                        detections.append([x1, y1, x2, y2])
                        confidences.append(score)
            
            # 缩小图像
            scale_factor *= self.scale
            new_w = int(gray.shape[1] / scale_factor)
            new_h = int(gray.shape[0] / scale_factor)
            if new_w < 64 or new_h < 128:
                break
            scaled_img = cv2.resize(gray, (new_w, new_h))
            level += 1
        
        if not detections:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        
        detections = np.array(detections, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)
        
        # Sigmoid归一化
        confidences = 1.0 / (1.0 + np.exp(-confidences))
        
        return detections, confidences
    
    def _filter_by_aspect_ratio(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """宽高比过滤"""
        if len(detections) == 0:
            return detections, confidences
        
        keep = []
        for i, det in enumerate(detections):
            w = det[2] - det[0]
            h = det[3] - det[1]
            if h <= 0:
                continue
            ratio = w / h
            
            if ratio < self.aspect_ratio_min:
                continue
            elif ratio > self.aspect_ratio_max:
                confidences[i] *= 0.5
                keep.append(i)
            else:
                keep.append(i)
        
        if not keep:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        
        return detections[keep], confidences[keep]
    
    def _split_merged_boxes(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """融合框拆分"""
        if len(detections) == 0:
            return detections, confidences
        
        new_detections = []
        new_confidences = []
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det
            w = x2 - x1
            h = y2 - y1
            
            if h <= 0:
                new_detections.append(det)
                new_confidences.append(confidences[i])
                continue
            
            ratio = w / h
            
            if ratio > self.aspect_ratio_max:
                half_w = w / 2
                
                new_detections.append([x1, y1, x1 + half_w, y2])
                new_confidences.append(confidences[i] * 0.8)
                
                new_detections.append([x2 - half_w, y1, x2, y2])
                new_confidences.append(confidences[i] * 0.8)
            else:
                new_detections.append(det)
                new_confidences.append(confidences[i])
        
        if not new_detections:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        
        return np.array(new_detections, dtype=np.float32), np.array(new_confidences, dtype=np.float32)
    
    @staticmethod
    def _nms(
        boxes: np.ndarray, 
        scores: np.ndarray, 
        threshold: float
    ) -> np.ndarray:
        """非极大值抑制"""
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
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(3780, dtype=np.float32)
        
        patch = image[y1:y2, x1:x2]
        patch = cv2.resize(patch, (64, 128))
        
        if self.use_hog_api:
            hog = cv2.HOGDescriptor(
                _winSize=(64, 128),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=9,
            )
            features = hog.compute(patch).flatten()
        else:
            features = self.hog_extractor.compute(patch)
        
        # L2归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
