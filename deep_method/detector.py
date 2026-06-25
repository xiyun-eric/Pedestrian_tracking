"""
YOLO11 检测器封装（深度学习版）

特性:
  - 支持标准 YOLOv8/v11 模型
  - 支持 LoRA 微调后的模型
  - 自动选择设备（CUDA/CPU）
  - 可配置的 NMS 后处理
  - 批量推理优化
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from typing import Tuple, Optional, List
import torch


class YOLODetector:
    """
    YOLO11 检测器封装
    
    支持标准推理和 LoRA 微调推理两种模式。
    对于 8GB 显存，推荐使用 yolo11s 或 yolo11m。
    """
    
    # YOLO 模型参数量参考
    MODEL_SIZES = {
        'yolo11n': (2.6, 'nano'),      # 2.6M 参数
        'yolo11s': (9.4, 'small'),     # 9.4M 参数
        'yolo11m': (20.1, 'medium'),   # 20.1M 参数
        'yolo11l': (25.3, 'large'),    # 25.3M 参数
        'yolov8n': (3.2, 'nano'),
        'yolov8s': (11.2, 'small'),
        'yolov8m': (25.9, 'medium'),
    }
    
    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        device: str = "cuda:0",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        classes: Optional[List[int]] = None,
    ):
        """
        Args:
            model_path: 模型路径或名称
            device: 设备 (cuda:0, cpu, mps)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
            max_det: 每帧最大检测数
            classes: 检测类别过滤 (None = 全部)
        """
        from ultralytics import YOLO
        
        # 检测设备
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[警告] CUDA 不可用，自动切换到 CPU")
            device = "cpu"
        
        self.device = device
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.classes = classes
        
        # 加载模型
        self.model = YOLO(model_path)
        
        # 模型信息
        self.model_name = Path(model_path).stem if Path(model_path).exists() else model_path
        
        if device.startswith("cuda"):
            print(f"[检测器] {self.model_name} | GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[检测器] {self.model_name} | CPU")
    
    def detect(
        self,
        image: np.ndarray,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行目标检测
        
        Args:
            image: BGR/RGB 图像 (H, W, 3)
            verbose: 是否打印详细信息
        
        Returns:
            detections: (N, 4) 边界框 [x1, y1, x2, y2]
            confidences: (N,) 置信度
            class_ids: (N,) 类别ID
        """
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            max_det=self.max_det,
            classes=self.classes,
            verbose=verbose,
        )
        
        detections = []
        confidences = []
        class_ids = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().item()
                cls = int(boxes.cls[i].cpu().item())
                
                detections.append(xyxy)
                confidences.append(conf)
                class_ids.append(cls)
        
        if detections:
            detections = np.array(detections, dtype=np.float32)
            confidences = np.array(confidences, dtype=np.float32)
            class_ids = np.array(class_ids, dtype=np.int32)
        else:
            detections = np.zeros((0, 4), dtype=np.float32)
            confidences = np.zeros((0,), dtype=np.float32)
            class_ids = np.zeros((0,), dtype=np.int32)
        
        return detections, confidences, class_ids
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        verbose: bool = False,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        批量检测（提升吞吐量）
        
        Args:
            images: 图像列表
        
        Returns:
            每张图的检测结果列表
        """
        results = self.model.predict(
            images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            max_det=self.max_det,
            classes=self.classes,
            verbose=verbose,
        )
        
        batch_results = []
        for result in results:
            detections = []
            confidences = []
            class_ids = []
            
            if result.boxes is not None:
                for i in range(len(result.boxes)):
                    detections.append(result.boxes.xyxy[i].cpu().numpy())
                    confidences.append(result.boxes.conf[i].cpu().item())
                    class_ids.append(int(result.boxes.cls[i].cpu().item()))
            
            if detections:
                detections = np.array(detections, dtype=np.float32)
                confidences = np.array(confidences, dtype=np.float32)
                class_ids = np.array(class_ids, dtype=np.int32)
            else:
                detections = np.zeros((0, 4), dtype=np.float32)
                confidences = np.zeros((0,), dtype=np.float32)
                class_ids = np.zeros((0,), dtype=np.int32)
            
            batch_results.append((detections, confidences, class_ids))
        
        return batch_results
    
    def warmup(self, img_size: Tuple[int, int] = (640, 640)):
        """预热模型（首次推理较慢）"""
        dummy = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
        _ = self.detect(dummy)
        print("[检测器] 预热完成")
    
    def load_lora(self, lora_path: str):
        """
        加载 LoRA 权重
        
        Args:
            lora_path: LoRA 权重文件路径
        """
        from model.lora_layers import load_lora_weights, inject_lora_to_yolo
        
        # 先融合 BN 层（必须在注入 LoRA 之前，否则 predict 时再次融合会冲突）
        try:
            self.model.model = self.model.model.fuse(verbose=False)
        except Exception:
            pass  # 已融合或无需融合
        
        # 注入 LoRA 结构
        self.model.model, _ = inject_lora_to_yolo(
            self.model.model, r=8, alpha=16.0, verbose=False
        )
        # 加载权重
        self.model.model = load_lora_weights(self.model.model, lora_path)
        print(f"[检测器] 已加载 LoRA 权重: {lora_path}")
    
    @property
    def info(self) -> dict:
        """获取检测器信息"""
        return {
            "model": self.model_name,
            "device": self.device,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
        }
