"""
传统方法跟踪可视化工具

从 code/Tracking/visualization.py 迁移，去除深度信息依赖。
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import colorsys


class TrackingVisualizer:
    """
    跟踪结果可视化工具

    功能:
    - 绘制检测框和跟踪框
    - 绘制跟踪轨迹
    - 绘制速度向量
    - 生成跟踪视频
    """

    def __init__(
        self,
        output_dir: str = "./tracking_output",
        line_thickness: int = 2,
        font_scale: float = 0.6,
        trajectory_length: int = 50,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.trajectory_length = trajectory_length

        self._color_cache: Dict[int, Tuple[int, int, int]] = {}

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """根据轨迹ID生成唯一颜色"""
        if track_id not in self._color_cache:
            hue = (track_id * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            self._color_cache[track_id] = bgr
        return self._color_cache[track_id]

    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        label: str = "",
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = None,
    ) -> np.ndarray:
        """绘制边界框"""
        if thickness is None:
            thickness = self.line_thickness

        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        if label:
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, thickness
            )

            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1,
            )

            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                thickness,
            )

        return image

    def draw_trajectory(
        self,
        image: np.ndarray,
        history: List[np.ndarray],
        color: Tuple[int, int, int] = (0, 255, 0),
        max_length: int = None,
    ) -> np.ndarray:
        """绘制跟踪轨迹"""
        if len(history) < 2:
            return image

        if max_length is None:
            max_length = self.trajectory_length

        recent_history = history[-max_length:]

        centers = []
        for bbox in recent_history:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((int(cx), int(cy)))

        for i in range(1, len(centers)):
            alpha = i / len(centers)
            thickness = max(1, int(self.line_thickness * alpha))
            cv2.line(image, centers[i - 1], centers[i], color, thickness)

        return image

    def draw_velocity_vector(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        velocity: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 255),
        scale: float = 5.0,
    ) -> np.ndarray:
        """绘制速度向量"""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        vx = velocity[0] * scale
        vy = velocity[1] * scale

        if abs(vx) < 1 and abs(vy) < 1:
            return image

        end_x = int(cx + vx)
        end_y = int(cy + vy)

        cv2.arrowedLine(
            image,
            (cx, cy),
            (end_x, end_y),
            color,
            self.line_thickness,
            cv2.LINE_AA,
            tipLength=0.3,
        )

        return image

    def draw_frame(
        self,
        image: np.ndarray,
        tracks: List,
        detections: List = None,
        draw_trajectory: bool = True,
        draw_velocity: bool = False,
        frame_id: int = None,
    ) -> np.ndarray:
        """绘制完整的一帧"""
        result = image.copy()

        if detections:
            for bbox, conf, cls_id in detections:
                label = f"Det {conf:.2f}"
                color = (128, 128, 128)
                result = self.draw_bbox(result, bbox, label, color, 1)

        for track in tracks:
            track_id = track.track_id
            bbox = track.get_bbox()
            confidence = track.confidence

            color = self.get_color(track_id)

            label = f"ID:{track_id} {confidence:.2f}"
            result = self.draw_bbox(result, bbox, label, color)

            if draw_trajectory and hasattr(track, 'history'):
                result = self.draw_trajectory(result, track.history, color)

            if draw_velocity and hasattr(track, 'get_velocity'):
                velocity = track.get_velocity()
                result = self.draw_velocity_vector(result, bbox, velocity, color)

        if frame_id is not None:
            cv2.putText(
                result,
                f"Frame: {frame_id}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                self.line_thickness,
            )

        return result

    def create_video_writer(
        self,
        output_name: str,
        fps: float,
        width: int,
        height: int,
    ) -> cv2.VideoWriter:
        """创建视频写入器"""
        output_path = self.output_dir / output_name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
