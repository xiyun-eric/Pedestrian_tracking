"""
传统方法卡尔曼滤波器

从 code/Tracking/kalman_filter.py 迁移，去除深度信息依赖，
8维状态向量: [x, y, w, h, vx, vy, vw, vh]
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


class KalmanFilter:
    """
    卡尔曼滤波器用于行人跟踪

    状态向量: [x, y, w, h, vx, vy, vw, vh]
    - (x, y): 边界框中心坐标
    - (w, h): 边界框宽高
    - (vx, vy, vw, vh): 对应的速度

    观测向量: [x, y, w, h]
    """

    def __init__(
        self,
        position_noise: float = 1.0,
        velocity_noise: float = 0.1,
    ):
        """
        初始化卡尔曼滤波器

        Args:
            position_noise: 位置观测噪声
            velocity_noise: 速度过程噪声
        """
        self.dim_x = 8
        self.dim_z = 4
        self.position_noise = position_noise
        self.velocity_noise = velocity_noise

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

        self.Q[self.dim_z:, self.dim_z:] *= self.velocity_noise

        self.R = np.eye(self.dim_z, dtype=np.float32)
        self.R[0, 0] = self.position_noise
        self.R[1, 1] = self.position_noise
        self.R[2, 2] = self.position_noise * 2
        self.R[3, 3] = self.position_noise * 2

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        初始化状态

        Args:
            measurement: [x, y, w, h] 边界框

        Returns:
            (state, covariance): 初始状态和协方差矩阵
        """
        mean = np.zeros(self.dim_x, dtype=np.float32)
        mean[:len(measurement)] = measurement

        std = []
        for i in range(self.dim_z):
            if i < 2:
                std.append(2 * measurement[2] if measurement[2] > 0 else 10)
            else:
                std.append(2 * measurement[i - 2] if measurement[i - 2] > 0 else 10)

        for i in range(self.dim_z):
            std.append(10 * std[i])

        covariance = np.diag(np.square(std)).astype(np.float32)

        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测下一状态
        """
        predicted_mean = self.F @ mean
        predicted_covariance = self.F @ covariance @ self.F.T + self.Q

        predicted_mean[2] = np.maximum(predicted_mean[2], 1.0)
        predicted_mean[3] = np.maximum(predicted_mean[3], 1.0)

        return predicted_mean, predicted_covariance

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用观测更新状态
        """
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

        return updated_mean, updated_covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将状态投影到观测空间
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
    ) -> float:
        """
        计算马氏距离
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurement = measurement[:2]

        innovation = measurement - projected_mean

        try:
            cholesky_factor = np.linalg.cholesky(projected_cov)
            d = np.sum(np.square(np.linalg.solve(cholesky_factor, innovation)))
        except np.linalg.LinAlgError:
            d = np.inf

        return float(d)

    @staticmethod
    def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """
        将边界框 [x1, y1, x2, y2] 转换为 [cx, cy, w, h]
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h], dtype=np.float32)

    @staticmethod
    def z_to_bbox(z: np.ndarray) -> np.ndarray:
        """
        将 [cx, cy, w, h] 转换为边界框 [x1, y1, x2, y2]
        """
        cx, cy, w, h = z[:4]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.array([x1, y1, x2, y2], dtype=np.float32)
