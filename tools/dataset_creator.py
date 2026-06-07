"""
自主采集数据集工具

功能:
  1. 视频录制 (摄像头采集)
  2. 视频分帧 (提取图像)
  3. 手动标注工具 (基于OpenCV)
  4. YOLO格式标签生成

使用方法:
  # 1. 采集视频
  python tools/dataset_creator.py --mode capture --output data/custom/videos --duration 60
  
  # 2. 视频分帧
  python tools/dataset_creator.py --mode extract --video data/custom/videos/scene1.mp4 --output data/custom/images/scene1 --target-fps 10 
    --split-ratio 0.67
  
  # 3. 标注 (需要图形界面)
  python tools/annotator.py --img-dir data/custom/images/scene1/train --output data/custom/annotations/scene1/train
  
  # 4. 转换格式
  python tools/dataset_creator.py --mode convert --ann-dir data/custom/annotations/scene1
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from typing import List, Tuple
import json


class VideoCapture:
    """摄像头视频采集"""
    
    def __init__(
        self,
        output_dir: Path,
        camera_id: int = 0,
        fps: int = 30,
        resolution: Tuple[int, int] = (1280, 720),
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
    
    def record(self, duration_sec: int = 60, scene_name: str = "scene") -> Path:
        """
        录制视频
        
        Args:
            duration_sec: 录制时长（秒）
            scene_name: 场景名称
        
        Returns:
            视频文件路径
        """
        cap = cv2.VideoCapture(self.camera_id)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_path = self.output_dir / f"{scene_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (actual_w, actual_h))
        
        print(f"开始录制: {scene_name}")
        print(f"分辨率: {actual_w}x{actual_h}, FPS: {self.fps}")
        print(f"录制时长: {duration_sec}秒")
        print("按 'q' 提前结束录制")
        
        total_frames = 0
        
        # 倒计时
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, f"Starting in {i}...", (actual_w//2-100, actual_h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            writer.write(frame)
        
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            remaining = duration_sec - elapsed
            
            # 显示录制信息
            cv2.putText(frame, f"Recording: {remaining:.0f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Frames: {total_frames}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            writer.write(frame)
            total_frames += 1
            
            cv2.imshow("Recording (Press 'q' to stop)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or elapsed >= duration_sec:
                break
        
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"录制完成: {video_path}")
        print(f"总帧数: {total_frames}")
        
        return video_path


class VideoExtractor:
    """视频分帧工具"""
    
    def __init__(self, frame_interval: int = 3, target_fps: int = None, split_ratio: float = None):
        """
        Args:
            frame_interval: 每隔多少帧保存一张（避免冗余）
            target_fps: 目标FPS，如果设置则自动计算帧间隔
            split_ratio: 训练集占比（如 0.67 表示前2/3为训练集），None表示不划分
        """
        self.frame_interval = frame_interval
        self.target_fps = target_fps
        self.split_ratio = split_ratio
    
    def extract(self, video_path: Path, output_dir: Path, prefix: str = "frame") -> int:
        """
        从视频提取帧，可选自动划分为训练集和验证集
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            prefix: 文件名前缀
        
        Returns:
            提取的帧数
        """
        # 创建目录结构
        if self.split_ratio is not None:
            train_dir = output_dir / "train"
            val_dir = output_dir / "val"
            train_dir.mkdir(parents=True, exist_ok=True)
            val_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 根据目标FPS计算帧间隔
        if self.target_fps is not None and fps > 0:
            self.frame_interval = max(1, int(fps / self.target_fps))
            print(f"视频信息: {total_frames}帧, {fps:.1f} FPS")
            print(f"目标FPS: {self.target_fps}, 计算帧间隔: {self.frame_interval}")
        else:
            print(f"视频信息: {total_frames}帧, {fps:.1f} FPS")
            print(f"每隔 {self.frame_interval} 帧保存一张")
        
        # 预计算提取的总帧数
        estimated_saved = total_frames // self.frame_interval + 1
        if self.split_ratio is not None:
            train_split = int(estimated_saved * self.split_ratio)
            print(f"自动划分: 前 {train_split} 张 -> 训练集, 后 {estimated_saved - train_split} 张 -> 验证集")
        
        count = 0
        saved = 0
        saved_train = 0
        saved_val = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % self.frame_interval == 0:
                if self.split_ratio is not None and saved >= train_split:
                    # 验证集
                    out_path = val_dir / f"{prefix}_{saved_val:06d}.png"
                    saved_val += 1
                else:
                    # 训练集或不划分
                    if self.split_ratio is not None:
                        out_path = train_dir / f"{prefix}_{saved_train:06d}.png"
                        saved_train += 1
                    else:
                        out_path = output_dir / f"{prefix}_{saved:06d}.png"
                
                cv2.imwrite(str(out_path), frame)
                saved += 1
            
            count += 1
            
            if count % 100 == 0:
                print(f"处理中: {count}/{total_frames}")
        
        cap.release()
        
        if self.split_ratio is not None:
            print(f"完成! 训练集: {saved_train} 张, 验证集: {saved_val} 张")
            print(f"训练集目录: {train_dir}")
            print(f"验证集目录: {val_dir}")
        else:
            print(f"完成! 保存了 {saved} 张图像到 {output_dir}")
        
        return saved


class FormatConverter:
    """标注格式转换工具"""
    
    def __init__(self, img_width: int = 1242, img_height: int = 375):
        self.img_width = img_width
        self.img_height = img_height
    
    def coco_to_yolo(self, bbox: List[float]) -> Tuple[float, float, float, float]:
        """
        COCO [x, y, w, h] -> YOLO [cx, cy, w, h] (归一化)
        """
        x, y, bw, bh = bbox
        cx = (x + bw / 2) / self.img_width
        cy = (y + bh / 2) / self.img_height
        w = bw / self.img_width
        h = bh / self.img_height
        return cx, cy, w, h
    
    def voc_to_yolo(self, bbox: List[float]) -> Tuple[float, float, float, float]:
        """
        VOC [xmin, ymin, xmax, ymax] -> YOLO [cx, cy, w, h] (归一化)
        """
        xmin, ymin, xmax, ymax = bbox
        cx = (xmin + xmax) / 2 / self.img_width
        cy = (ymin + ymax) / 2 / self.img_height
        w = (xmax - xmin) / self.img_width
        h = (ymax - ymin) / self.img_height
        return cx, cy, w, h
    
    def yolo_to_voc(self, label: str) -> Tuple[int, int, int, int, int]:
        """
        YOLO [class, cx, cy, w, h] -> VOC [class, xmin, ymin, xmax, ymax]
        """
        parts = label.strip().split()
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        
        xmin = int((cx - bw / 2) * self.img_width)
        ymin = int((cy - bh / 2) * self.img_height)
        xmax = int((cx + bw / 2) * self.img_width)
        ymax = int((cy + bh / 2) * self.img_height)
        
        return cls_id, xmin, ymin, xmax, ymax


def main():
    parser = argparse.ArgumentParser(description="自主数据集采集工具")
    parser.add_argument("--mode", choices=["capture", "extract", "convert"],
                       default="capture", help="操作模式")
    
    # 采集参数
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=int, default=60,
                       help="录制时长(秒)")
    parser.add_argument("--scene", type=str, default="scene",
                       help="场景名称")
    parser.add_argument("--fps", type=int, default=30)
    
    # 提取参数
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--interval", type=int, default=3,
                       help="帧间隔（与 --target-fps 二选一）")
    parser.add_argument("--target-fps", type=int, default=None,
                       help="目标FPS，自动计算帧间隔（优先于 --interval）")
    parser.add_argument("--split-ratio", type=float, default=None,
                       help="训练集占比，如 0.67 表示前2/3为训练集，后1/3为验证集")
    
    # 通用
    parser.add_argument("--output", type=str, default="data/custom")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.mode == "capture":
        print("=" * 50)
        print("  摄像头视频采集")
        print("=" * 50)
        
        capture = VideoCapture(
            output_dir / "videos",
            camera_id=args.camera,
            fps=args.fps,
        )
        video_path = capture.record(
            duration_sec=args.duration,
            scene_name=args.scene,
        )
        
        print(f"\n下一步: 提取帧")
        print(f"  python tools/dataset_creator.py --mode extract "
              f"--video {video_path} --output {output_dir}/images/{args.scene}")
    
    elif args.mode == "extract":
        if not args.video:
            print("错误: 请指定 --video 参数")
            return
        
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"错误: 视频文件不存在: {video_path}")
            return
        
        # 直接使用 --output 指定的目录作为输出目录
        output_img_dir = output_dir
        
        # 优先使用 target_fps，否则使用 interval
        extractor = VideoExtractor(
            frame_interval=args.interval,
            target_fps=args.target_fps,
            split_ratio=args.split_ratio
        )
        extractor.extract(video_path, output_img_dir)
        
        print(f"\n下一步: 标注")
        print(f"  python tools/annotator.py --img-dir {output_img_dir} "
              f"--output {output_dir}/annotations/{video_path.stem}")
    
    elif args.mode == "convert":
        print("格式转换 (TODO)")


if __name__ == "__main__":
    main()
