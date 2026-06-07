"""
卡尔曼滤波器用于行人跟踪（支持深度信息）

状态向量: [x, y, w, h, d, vx, vy, vw, vh, vd]
- (x, y): 边界框中心坐标
- (w, h): 边界框宽高
- d: 深度值
- (vx, vy, vw, vh, vd): 对应的速度

观测向量: [x, y, w, h, d]
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


class KalmanFilter:
    """
    卡尔曼滤波器用于行人跟踪（支持深度信息）
    
    状态向量: [x, y, w, h, d, vx, vy, vw, vh, vd]
    - (x, y): 边界框中心坐标
    - (w, h): 边界框宽高
    - d: 深度值
    - (vx, vy, vw, vh, vd): 对应的速度
    
    观测向量: [x, y, w, h, d]
    """
    
    def __init__(
        self, 
        use_depth: bool = True,
        depth_scale: float = 80.0,
        position_noise: float = 1.0,
        velocity_noise: float = 0.1,
        depth_noise: float = 2.0,
    ):
        """
        初始化卡尔曼滤波器
        
        Args:
            use_depth: 是否使用深度信息
            depth_scale: 深度归一化尺度（米）
            position_noise: 位置观测噪声
            velocity_noise: 速度过程噪声
            depth_noise: 深度观测噪声
        """
        self.use_depth = use_depth
        self.depth_scale = depth_scale
        self.position_noise = position_noise
        self.velocity_noise = velocity_noise
        self.depth_noise = depth_noise
        
        if self.use_depth:
            self.dim_x = 10
            self.dim_z = 5
        else:
            self.dim_x = 8
            self.dim_z = 4
        
        self._init_matrices()
    
    def _init_matrices(self):
        self.F = np.eye(self.dim_x, dtype=np.float32)
        for i in range(self.dim_z):
            self.F[i, i + self.dim_z] = 1
        
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        for i in range(self.dim_z):
            self.H[i, i] = 1
        
        self._init_noise_matrices()
    
    def _init_noise_matrices(self):
        self.Q = np.eye(self.dim_x, dtype=np.float32)
        
        self.Q[0, 0] = self.position_noise
        self.Q[1, 1] = self.position_noise
        self.Q[2, 2] = self.position_noise * 0.5
        self.Q[3, 3] = self.position_noise * 0.5
        
        if self.use_depth:
            self.Q[4, 4] = self.depth_noise
        
        self.Q[self.dim_z:, self.dim_z:] *= self.velocity_noise
        
        self.R = np.eye(self.dim_z, dtype=np.float32)
        self.R[0, 0] = self.position_noise
        self.R[1, 1] = self.position_noise
        self.R[2, 2] = self.position_noise * 2
        self.R[3, 3] = self.position_noise * 2
        
        if self.use_depth:
            self.R[4, 4] = self.depth_noise * 2
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        初始化状态
        
        Args:
            measurement: [x, y, w, h] 或 [x, y, w, h, d] 边界框和深度
        
        Returns:
            (state, covariance): 初始状态和协方差矩阵
        """
        mean = np.zeros(self.dim_x, dtype=np.float32)
        mean[:len(measurement)] = measurement
        
        if self.use_depth and len(measurement) == 4:
            measurement = np.append(measurement, 0.0)
        
        std = []
        for i in range(self.dim_z):
            if i < 2:
                std.append(2 * measurement[2] if measurement[2] > 0 else 10)
            elif i < 4:
                std.append(2 * measurement[i - 2] if measurement[i - 2] > 0 else 10)
            else:
                std.append(self.depth_scale * 0.1)
        
        for i in range(self.dim_z):
            std.append(10 * std[i])
        
        covariance = np.diag(np.square(std)).astype(np.float32)
        
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测下一状态
        
        Args:
            mean: 当前状态均值
            covariance: 当前状态协方差
        
        Returns:
            (predicted_mean, predicted_covariance): 预测的状态和协方差
        """
        predicted_mean = self.F @ mean
        predicted_covariance = self.F @ covariance @ self.F.T + self.Q
        
        predicted_mean[2] = np.maximum(predicted_mean[2], 1.0)
        predicted_mean[3] = np.maximum(predicted_mean[3], 1.0)
        
        if self.use_depth and len(predicted_mean) > 4:
            predicted_mean[4] = np.maximum(predicted_mean[4], 0.0)
        
        return predicted_mean, predicted_covariance
    
    def update(self, mean: np.ndarray, covariance: np.ndarray, 
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用观测更新状态
        
        Args:
            mean: 预测状态均值
            covariance: 预测状态协方差
            measurement: 观测值 [x, y, w, h] 或 [x, y, w, h, d]
        
        Returns:
            (updated_mean, updated_covariance): 更新后的状态和协方差
        """
        if self.use_depth and len(measurement) == 4:
            measurement = np.append(measurement, 0.0)
        
        projected_mean = self.H @ mean
        innovation = measurement - projected_mean
        
        innovation_covariance = self.H @ covariance @ self.H.T + self.R
        
        try:
            kalman_gain = covariance @ self.H.T @ np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            kalman_gain = np.zeros_like(covariance @ self.H.T)
        
        updated_mean = mean + kalman_gain @ innovation
        updated_covariance = (np.eye(self.dim_x) - kalman_gain @ self.H) @ covariance
        
        updated_mean[2] = np.maximum(updated_mean[2], 1.0)
        updated_mean[3] = np.maximum(updated_mean[3], 1.0)
        
        if self.use_depth and len(updated_mean) > 4:
            updated_mean[4] = np.maximum(updated_mean[4], 0.0)
        
        return updated_mean, updated_covariance
    
    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将状态投影到观测空间
        
        Args:
            mean: 状态均值
            covariance: 状态协方差
        
        Returns:
            (projected_mean, projected_covariance): 投影后的均值和协方差
        """
        projected_mean = self.H @ mean
        projected_covariance = self.H @ covariance @ self.H.T + self.R
        
        return projected_mean, projected_covariance
    
    def mahalanobis_distance(
        self, 
        mean: np.ndarray, 
        covariance: np.ndarray, 
        measurement: np.ndarray,
        only_position: bool = False,
        depth_weight: float = 0.5,
    ) -> float:
        """
        计算马氏距离
        
        Args:
            mean: 状态均值
            covariance: 状态协方差
            measurement: 观测值
            only_position: 是否只计算位置距离
            depth_weight: 深度信息的权重（0-1）
        
        Returns:
            马氏距离
        """
        if self.use_depth and len(measurement) == 4:
            measurement = np.append(measurement, 0.0)
        
        projected_mean, projected_cov = self.project(mean, covariance)
        
        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurement = measurement[:2]
        elif self.use_depth and len(projected_cov) > 4:
            projected_cov[4, 4] *= (1.0 / max(depth_weight, 0.01))
        
        innovation = measurement - projected_mean
        
        try:
            cholesky_factor = np.linalg.cholesky(projected_cov)
            d = np.sum(np.square(np.linalg.solve(cholesky_factor, innovation)))
        except np.linalg.LinAlgError:
            d = np.inf
        
        return float(d)
    
    @staticmethod
    def bbox_to_z(bbox: np.ndarray, depth: Optional[float] = None) -> np.ndarray:
        """
        将边界框 [x1, y1, x2, y2] 转换为 [cx, cy, w, h] 或 [cx, cy, w, h, d]
        
        Args:
            bbox: [x1, y1, x2, y2] 左上角和右下角坐标
            depth: 深度值（可选）
        
        Returns:
            [cx, cy, w, h] 或 [cx, cy, w, h, d]: 中心坐标、宽高和深度
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        if depth is not None:
            return np.array([cx, cy, w, h, depth], dtype=np.float32)
        return np.array([cx, cy, w, h], dtype=np.float32)
    
    @staticmethod
    def z_to_bbox(z: np.ndarray) -> np.ndarray:
        """
        将 [cx, cy, w, h] 或 [cx, cy, w, h, d] 转换为边界框 [x1, y1, x2, y2]
        
        Args:
            z: [cx, cy, w, h] 或 [cx, cy, w, h, d] 中心坐标、宽高和深度
        
        Returns:
            [x1, y1, x2, y2]: 左上角和右下角坐标
        """
        cx, cy, w, h = z[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    @staticmethod
    def get_depth(z: np.ndarray) -> Optional[float]:
        """
        从状态向量中获取深度值
        
        Args:
            z: 状态向量
        
        Returns:
            深度值或None
        """
        if len(z) >= 5:
            return float(z[4])
        return None