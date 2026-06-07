"""
深度学习多目标跟踪管道 (YOLO11 + AdvancedTracker)

完整流程:
  1. YOLO11 高精度行人检测
  2. 卡尔曼滤波运动预测
  3. 级联匹配（马氏距离 + IoU + 外观特征）
  4. ByteTrack 风格的低分框二次匹配
  5. 匈牙利算法最优分配
  6. 轨迹管理和输出
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import time
from dataclasses import dataclass

# 导入跟踪模块
from deep_method.tracking import (
    AdvancedTracker,
    AdvancedTrackerConfig,
    ReIDExtractor,
    TrackerConfig,
    get_config,
    create_tracker,
)

from detector import YOLODetector


@dataclass
class DeepTrackingConfig:
    """深度学习跟踪配置"""
    # 检测参数
    model_path: str = "yolo11m.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cuda:0"
    
    # ByteTrack 低分框
    use_bytetrack: bool = True
    low_conf_threshold: float = 0.1
    
    # 跟踪参数
    tracker_type: str = "advanced"  # advanced, ocsort, strongsort
    preset: str = "standard"  # fast, standard, high_precision, crowded_scene, highway
    max_age: int = 30
    min_hits: int = 3
    track_iou_threshold: float = 0.3
    
    # ReID 参数
    use_reid: bool = True
    reid_model: str = "osnet_x1_0"
    
    # LoRA
    lora_path: Optional[str] = None
    
    # 输出参数
    output_dir: str = "outputs/deep"
    save_video: bool = True
    save_frames: bool = True
    frame_skip: int = 10
    
    # 可视化
    draw_trajectory: bool = True
    draw_prediction: bool = False


class DeepTrackingPipeline:
    """
    深度学习方法多目标跟踪主管道
    
    整合:
    - YOLO11 检测器（支持 LoRA 微调）
    - ByteTrack 低分框策略
    - 卡尔曼滤波 + 级联匹配 + 匈牙利算法
    - 外观特征 ReID（可选）
    - AdvancedTracker 增强版跟踪器
    """
    
    def __init__(self, config: DeepTrackingConfig = None):
        self.config = config or DeepTrackingConfig()
        
        # 初始化检测器
        self.detector = YOLODetector(
            model_path=self.config.model_path,
            device=self.config.device,
            conf_threshold=self.config.conf_threshold,
            iou_threshold=self.config.iou_threshold,
            classes=[0],  # 只检测 person 类
        )
        
        # 加载 LoRA 权重
        if self.config.lora_path:
            self.detector.load_lora(self.config.lora_path)
        
        # 初始化跟踪器
        self._init_tracker()
        
        # 低分框检测器（ByteTrack策略）
        if self.config.use_bytetrack:
            self.low_conf_detector = YOLODetector(
                model_path=self.config.model_path,
                device=self.config.device,
                conf_threshold=self.config.low_conf_threshold,
                iou_threshold=self.config.iou_threshold,
                max_det=500,
                classes=[0],
            )
            if self.config.lora_path:
                self.low_conf_detector.load_lora(self.config.lora_path)
        else:
            self.low_conf_detector = None
        
        # ReID 特征提取器
        if self.config.use_reid:
            self.reid_extractor = ReIDExtractor(
                model_name=self.config.reid_model,
                device=self.config.device,
            )
        else:
            self.reid_extractor = None
        
        # 可视化器
        self.visualizer = SimpleVisualizer()
        
        # 统计
        self.frame_count = 0
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "max_tracks": 0,
            "processing_time": 0.0,
            "high_conf_dets": 0,
            "low_conf_dets": 0,
            "reid_features": 0,
        }
    
    def _init_tracker(self):
        """初始化跟踪器"""
        tracker_config = get_config(self.config.preset)
        tracker_config.max_age = self.config.max_age
        tracker_config.min_hits = self.config.min_hits
        tracker_config.iou_threshold = self.config.track_iou_threshold
        tracker_config.use_reid = self.config.use_reid
        tracker_config.reid_model = self.config.reid_model

        if self.config.tracker_type == "advanced":
            self.tracker = AdvancedTracker(tracker_config)
        else:
            raise ValueError(f"未知跟踪器类型: {self.config.tracker_type}（仅支持 advanced）")

        print(f"[Pipeline] 跟踪器类型: {self.config.tracker_type}, 预设: {self.config.preset}")
    
    def process_sequence(
        self,
        image_dir: Path,
        output_dir: Path,
        max_frames: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """处理一个图像序列"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取帧列表
        frame_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        if not frame_files:
            print(f"错误: {image_dir} 中未找到图像")
            return {"error": "no_images"}
        
        first_frame = cv2.imread(str(frame_files[0]))
        h, w = first_frame.shape[:2]
        
        # 视频写入器
        video_writer = None
        if self.config.save_video:
            video_path = output_dir / "tracking_result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))
        
        if self.config.save_frames:
            (output_dir / "frames").mkdir(exist_ok=True)
        
        # 预热
        if self.frame_count == 0:
            self.detector.warmup(img_size=(w, h))
        
        self.stats["total_frames"] = len(frame_files)
        
        print(f"\n{'='*60}")
        print(f"深度学习方法跟踪: {image_dir.name}")
        print(f"模型: {self.config.model_path} | 设备: {self.config.device}")
        print(f"跟踪器: {self.config.tracker_type} | ReID: {self.config.use_reid}")
        print(f"总帧数: {len(frame_files)}, 尺寸: {w}x{h}")
        print(f"{'='*60}")
        
        confirmed_tracks = []

        for frame_idx, frame_path in enumerate(frame_files):
            t_start = time.time()

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            # === 高置信度检测 ===
            detections, confidences, class_ids = self.detector.detect(frame)
            self.stats["high_conf_dets"] += len(detections)
            
            # === ByteTrack 低分框 ===
            low_detections = None
            low_confidences = None
            
            if self.config.use_bytetrack and self.low_conf_detector:
                low_dets, low_confs, low_cls = self.low_conf_detector.detect(frame)
                
                # 过滤：只保留低分框
                low_mask = low_confs < self.config.conf_threshold
                if np.any(low_mask):
                    low_detections = low_dets[low_mask]
                    low_confidences = low_confs[low_mask]
                    self.stats["low_conf_dets"] += len(low_detections)
            
            # === ReID 特征提取 ===
            features = None
            if self.config.use_reid and self.reid_extractor is not None and len(detections) > 0:
                features = self.reid_extractor.extract_features_batch(
                    frame, detections
                )
                self.stats["reid_features"] += len(detections)
            
            # === 跟踪更新 ===
            if len(detections) > 0 or self.config.tracker_type != "advanced":
                # 使用新跟踪器
                if self.config.tracker_type == "advanced":
                    confirmed_tracks = self.tracker.update(
                        detections,
                        confidences,
                        features=features,
                        image=frame,
                        low_conf_detections=low_detections,
                        low_conf_confidences=low_confidences,
                    )
                else:
                    confirmed_tracks = self.tracker.update(
                        detections,
                        confidences,
                        features=features,
                        image=frame,
                    )
            else:
                confirmed_tracks = []
            
            self.stats["total_detections"] += len(detections)
            self.stats["max_tracks"] = max(self.stats["max_tracks"], len(confirmed_tracks))
            
            # === 可视化 ===
            vis_image = self._visualize(frame, detections, confidences, confirmed_tracks, frame_idx)
            
            if video_writer:
                video_writer.write(vis_image)
            
            if self.config.save_frames and frame_idx % self.config.frame_skip == 0:
                cv2.imwrite(str(output_dir / "frames" / f"frame_{frame_idx:06d}.jpg"), vis_image)
            
            self.stats["processing_time"] += time.time() - t_start
            self.frame_count += 1
            
            if verbose and (frame_idx + 1) % 10 == 0:
                fps = (frame_idx + 1) / max(self.stats["processing_time"], 0.001)
                print(f"  帧 {frame_idx+1}/{len(frame_files)} | "
                      f"检测: {len(detections)} | 跟踪: {len(confirmed_tracks)} | "
                      f"FPS: {fps:.1f}")
            
            # 定期清理显存
            if frame_idx % 100 == 0:
                import torch
                torch.cuda.empty_cache()
        
        if video_writer:
            video_writer.release()
        
        avg_fps = len(frame_files) / max(self.stats["processing_time"], 0.001)
        print(f"\n完成! 平均 FPS: {avg_fps:.1f}")
        print(f"最大同时跟踪目标: {self.stats['max_tracks']}")
        print(f"ReID 特征提取次数: {self.stats['reid_features']}")
        print(f"结果保存在: {output_dir}")
        
        return self.stats
    
    def _visualize(
        self,
        frame: np.ndarray,
        detections: np.ndarray,
        confidences: np.ndarray,
        tracks: List,
        frame_idx: int,
    ) -> np.ndarray:
        """可视化"""
        vis_image = frame.copy()
        h, w = vis_image.shape[:2]
        
        # 绘制检测框（灰色）
        for det, conf in zip(detections, confidences):
            x1, y1, x2, y2 = map(int, det)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (100, 100, 100), 1)
            cv2.putText(vis_image, f"{conf:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        # 绘制跟踪结果
        for track in tracks:
            tid = track.track_id
            bbox = track.get_bbox()
            color = self.visualizer.get_color(tid)
            
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            label = f"ID:{tid}"
            if hasattr(track, 'confidence'):
                label += f" C:{track.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_image, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(vis_image, label, (x1+2, y1-3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 绘制轨迹
            if self.config.draw_trajectory and hasattr(track, 'history') and len(track.history) > 1:
                history = track.history[-30:]
                for i in range(1, len(history)):
                    cx1 = int((history[i-1][0] + history[i-1][2]) / 2)
                    cy1 = int((history[i-1][1] + history[i-1][3]) / 2)
                    cx2 = int((history[i][0] + history[i][2]) / 2)
                    cy2 = int((history[i][1] + history[i][3]) / 2)
                    alpha = i / len(history)
                    cv2.line(vis_image, (cx1, cy1), (cx2, cy2), color, max(1, int(2*alpha)))
        
        # 帧信息
        info_text = f"Frame: {frame_idx+1} | Tracks: {len(tracks)} | Dets: {len(detections)}"
        cv2.putText(vis_image, info_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 方法标识
        method_label = f"{self.config.tracker_type.upper()} + YOLO11"
        if self.config.use_reid:
            method_label += f" + {self.config.reid_model}"
        cv2.putText(vis_image, method_label, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return vis_image
    
    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        max_frames: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        处理视频文件

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            max_frames: 最大处理帧数
            verbose: 是否打印详细信息

        Returns:
            统计信息字典
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            return {"error": "cannot_open_video"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames is None or max_frames > total_frames:
            max_frames = total_frames

        # 视频写入器
        video_writer = None
        if self.config.save_video:
            video_path_out = output_dir / "tracking_result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path_out), fourcc, int(fps), (width, height))

        if self.config.save_frames:
            (output_dir / "frames").mkdir(exist_ok=True)

        # 预热
        if self.frame_count == 0:
            self.detector.warmup(img_size=(width, height))

        self.stats["total_frames"] = max_frames

        print(f"\n{'='*60}")
        print(f"深度学习方法跟踪: {video_path.name}")
        print(f"模型: {self.config.model_path} | 设备: {self.config.device}")
        print(f"跟踪器: {self.config.tracker_type} | ReID: {self.config.use_reid}")
        print(f"视频FPS: {fps}, 总帧数: {total_frames}, 尺寸: {width}x{height}")
        print(f"处理帧数: {max_frames}")
        print(f"{'='*60}")

        confirmed_tracks = []

        for frame_idx in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.time()

            # 高置信度检测
            detections, confidences, class_ids = self.detector.detect(frame)
            self.stats["high_conf_dets"] += len(detections)

            # ByteTrack 低分框
            low_detections = None
            low_confidences = None

            if self.config.use_bytetrack and self.low_conf_detector:
                low_dets, low_confs, low_cls = self.low_conf_detector.detect(frame)

                low_mask = low_confs < self.config.conf_threshold
                if np.any(low_mask):
                    low_detections = low_dets[low_mask]
                    low_confidences = low_confs[low_mask]
                    self.stats["low_conf_dets"] += len(low_detections)

            # ReID 特征提取
            features = None
            if self.config.use_reid and self.reid_extractor is not None and len(detections) > 0:
                features = self.reid_extractor.extract_features_batch(
                    frame, detections
                )
                self.stats["reid_features"] += len(detections)

            # 跟踪更新
            if len(detections) > 0:
                confirmed_tracks = self.tracker.update(
                    detections,
                    confidences,
                    features=features,
                    image=frame,
                    low_conf_detections=low_detections,
                    low_conf_confidences=low_confidences,
                )
            else:
                confirmed_tracks = []

            self.stats["total_detections"] += len(detections)
            self.stats["max_tracks"] = max(self.stats["max_tracks"], len(confirmed_tracks))

            # 可视化
            vis_image = self._visualize(frame, detections, confidences, confirmed_tracks, frame_idx)

            if video_writer:
                video_writer.write(vis_image)

            if self.config.save_frames and frame_idx % self.config.frame_skip == 0:
                cv2.imwrite(str(output_dir / "frames" / f"frame_{frame_idx:06d}.jpg"), vis_image)

            self.stats["processing_time"] += time.time() - t_start
            self.frame_count += 1

            if verbose and (frame_idx + 1) % 10 == 0:
                current_fps = (frame_idx + 1) / max(self.stats["processing_time"], 0.001)
                print(f"  帧 {frame_idx+1}/{max_frames} | "
                      f"检测: {len(detections)} | 跟踪: {len(confirmed_tracks)} | "
                      f"FPS: {current_fps:.1f}")

            # 定期清理显存
            if frame_idx % 100 == 0:
                import torch
                torch.cuda.empty_cache()

        cap.release()
        if video_writer:
            video_writer.release()

        avg_fps = self.stats["total_frames"] / max(self.stats["processing_time"], 0.001)
        print(f"\n完成! 平均 FPS: {avg_fps:.1f}")
        print(f"最大同时跟踪目标: {self.stats['max_tracks']}")
        print(f"ReID 特征提取次数: {self.stats['reid_features']}")
        print(f"结果保存在: {output_dir}")

        return self.stats

    def reset(self):
        """重置状态"""
        self.tracker.reset()
        if self.reid_extractor:
            self.reid_extractor.clear_cache()
        self.frame_count = 0
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "max_tracks": 0,
            "processing_time": 0.0,
            "high_conf_dets": 0,
            "low_conf_dets": 0,
            "reid_features": 0,
        }
    
    def get_tracker_info(self) -> Dict:
        """获取跟踪器信息"""
        return self.tracker.get_tracking_info()


class SimpleVisualizer:
    """简单可视化器（兼容替代）"""
    
    def __init__(self):
        self.colors = {}
    
    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """获取颜色"""
        if track_id not in self.colors:
            # 生成随机颜色
            np.random.seed(track_id)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            self.colors[track_id] = color
        return self.colors[track_id]


# 便捷函数
def run_tracking(
    image_dir: str,
    output_dir: str,
    model_path: str = "yolo11m.pt",
    tracker_type: str = "advanced",
    preset: str = "standard",
    use_reid: bool = True,
    device: str = "cuda:0",
    max_frames: Optional[int] = None,
) -> Dict:
    """
    快速运行跟踪
    
    Args:
        image_dir: 图像目录
        output_dir: 输出目录
        model_path: YOLO 模型路径
        tracker_type: 跟踪器类型
        preset: 配置预设
        use_reid: 是否使用 ReID
        device: 设备
        max_frames: 最大帧数
    
    Returns:
        统计信息
    """
    config = DeepTrackingConfig(
        model_path=model_path,
        tracker_type=tracker_type,
        preset=preset,
        use_reid=use_reid,
        device=device,
        output_dir=output_dir,
    )
    
    pipeline = DeepTrackingPipeline(config)
    
    return pipeline.process_sequence(
        Path(image_dir),
        Path(output_dir),
        max_frames=max_frames,
    )