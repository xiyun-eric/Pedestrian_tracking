"""
传统CV方法多目标跟踪管道

完整流程:
  1. HOG + SVM 行人检测
  2. Farneback 稠密光流运动估计
  3. 卡尔曼滤波状态预测与更新
  4. 级联匹配（马氏距离 + IoU 代价矩阵）
  5. 匈牙利算法最优分配
  6. 颜色直方图 + HOG 特征 ReID（遮挡处理）
  7. 轨迹管理（创建/确认/删除）
  8. 结果可视化与输出

核心算法说明:
  - 卡尔曼滤波: 8维状态 [x,y,w,h,vx,vy,vw,vh]，匀速运动模型
  - 级联匹配: 按丢失时间从小到大匹配，优先给"近期丢失"的轨迹机会
  - 匈牙利算法: 在代价矩阵上寻找最小代价的二分图匹配
  - 马氏距离: 考虑状态不确定性的统计距离
  - IoU距离: 1 - IoU，用于空间重叠度量
"""

import sys
import time
from pathlib import Path
from tqdm import tqdm

# 添加 common 模块路径
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import time
from dataclasses import dataclass

# 导入跟踪模块（已从 code/Tracking 迁移至 traditional_method/）
from traditional_method.kalman_filter import KalmanFilter
from traditional_method.tracker import Tracker, Track
from traditional_method.visualization import TrackingVisualizer

# 导入新增的传统CV模块
from traditional_method.hog_detector import HOGDetector
from traditional_method.feature_extraction import FeatureExtractor, ReIDMatcher, FeatureConfig


@dataclass
class TraditionalTrackingConfig:
    """传统方法跟踪配置"""
    # 检测参数
    hog_win_stride: Tuple[int, int] = (8, 8)
    hog_scale: float = 1.05
    hog_conf_threshold: float = 0.3  # 置信度阈值（平衡召回率与精确率）
    use_hog_api: bool = True         # HOG特征提取是否使用OpenCV API（默认True，速度快）
    use_svm_api: bool = True         # SVM是否使用OpenCV预训练权重（默认True）
    
    # 跟踪参数
    max_age: int = 15             # 轨迹最大丢失帧数（缩短以减少幽灵框）
    min_hits: int = 3             # 确认轨迹所需最小检测数
    iou_threshold: float = 0.3    # IoU 匹配阈值
    
    # 光流参数
    use_optical_flow: bool = True
    flow_pyr_scale: float = 0.5
    flow_levels: int = 3
    flow_winsize: int = 15
    flow_iterations: int = 3
    flow_correction_weight: float = 0.5  # 光流校正权重（增大以更好应对变速运动）
    
    # ReID参数
    use_reid: bool = True
    reid_high_threshold: float = 0.85
    reid_low_threshold: float = 0.5
    
    # 输出参数
    output_dir: str = "outputs/traditional"
    save_video: bool = True
    save_frames: bool = True
    frame_skip: int = 10         # 每10帧保存一张图


class OpticalFlowEstimator:
    """
    Farneback 稠密光流估计器
    
    原理:
    - 使用多项式展开近似每个像素的邻域
    - 通过最小化相邻帧的像素误差来估计运动向量
    - 稠密光流: 对每个像素都计算运动向量
    
    用途:
    - 辅助卡尔曼滤波进行运动预测
    - 检测运动一致性（运动方向突变的可能是错误匹配）
    - 辅助遮挡检测（运动向量为零可能是静止遮挡物）
    """
    
    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        
        self._prev_gray = None
    
    def compute(
        self, 
        current_gray: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        计算当前帧与前一帧之间的稠密光流
        
        Args:
            current_gray: 当前帧灰度图
        
        Returns:
            flow: (H, W, 2) 光流场 [dy, dx]，第一帧返回 None
        """
        if self._prev_gray is None:
            self._prev_gray = current_gray.copy()
            return None
        
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, current_gray,
            None,  # 不传入初始流
            self.pyr_scale,
            self.levels,
            self.winsize,
            self.iterations,
            self.poly_n,
            self.poly_sigma,
            0,  # 不使用高斯滤波标志
        )
        
        self._prev_gray = current_gray.copy()
        return flow
    
    def get_bbox_motion(
        self, 
        flow: np.ndarray, 
        bbox: np.ndarray
    ) -> Tuple[float, float]:
        """
        从光流场中提取指定边界框的平均运动
        
        Args:
            flow: 光流场
            bbox: [x1, y1, x2, y2]
        
        Returns:
            (mean_dx, mean_dy): 平均位移向量
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(flow.shape[1], x2)
        y2 = min(flow.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0
        
        patch = flow[y1:y2, x1:x2]
        mean_dx = np.mean(patch[..., 0])
        mean_dy = np.mean(patch[..., 1])
        
        return float(mean_dx), float(mean_dy)
    
    def reset(self):
        """重置状态"""
        self._prev_gray = None


class TraditionalTrackingPipeline:
    """
    传统CV方法多目标跟踪主管道
    
    整合以下模块:
    - HOG + SVM 行人检测（HOGDetector）
    - Farneback 光流运动估计（OpticalFlowEstimator）
    - 卡尔曼滤波 + 级联匹配 + 匈牙利算法（Tracker from code/Tracking）
    - 颜色直方图 + HOG 特征 ReID（FeatureExtractor + ReIDMatcher）
    - 轨迹可视化（TrackingVisualizer from code/Tracking）
    """
    
    def __init__(self, config: TraditionalTrackingConfig = None):
        self.config = config or TraditionalTrackingConfig()
        
        # 初始化各模块
        self.detector = HOGDetector(
            win_stride=self.config.hog_win_stride,
            scale=self.config.hog_scale,
            conf_threshold=self.config.hog_conf_threshold,
            use_hog_api=self.config.use_hog_api,
            use_svm_api=self.config.use_svm_api,
        )
        
        self.flow_estimator = OpticalFlowEstimator(
            pyr_scale=self.config.flow_pyr_scale,
            levels=self.config.flow_levels,
            winsize=self.config.flow_winsize,
            iterations=self.config.flow_iterations,
        )
        
        self.tracker = Tracker(
            iou_threshold=self.config.iou_threshold,
            max_age=self.config.max_age,
            min_hits=self.config.min_hits,
            flow_correction_weight=self.config.flow_correction_weight,
        )
        
        if self.config.use_reid:
            feat_config = FeatureConfig()
            self.feature_extractor = FeatureExtractor(feat_config, use_opencv_api=self.config.use_hog_api)
            self.reid_matcher = ReIDMatcher(
                self.feature_extractor,
                high_threshold=self.config.reid_high_threshold,
                low_threshold=self.config.reid_low_threshold,
            )
        else:
            self.feature_extractor = None
            self.reid_matcher = None
        
    def process_sequence(
        self,
        image_dir: Path,
        output_dir: Path,
        max_frames: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        处理一个图像序列
        
        Args:
            image_dir: 图像目录
            output_dir: 输出目录
            max_frames: 最大处理帧数
            verbose: 是否打印详细信息
        
        Returns:
            统计信息字典
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取帧列表
        frame_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
        if max_frames:
            frame_files = frame_files[:max_frames]
        
        if not frame_files:
            print(f"错误: {image_dir} 中未找到图像")
            return {"error": "no_images"}
        
        # 初始化输出
        first_frame = cv2.imread(str(frame_files[0]))
        if first_frame is None:
            print(f"错误: 无法读取第一帧")
            return {"error": "cannot_read"}
        
        h, w = first_frame.shape[:2]
        video_writer = None
        
        if self.config.save_video:
            video_path = output_dir / "tracking_result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))
        
        if self.config.save_frames:
            (output_dir / "frames").mkdir(exist_ok=True)
        
        # 统计信息
        stats = {
            "total_frames": len(frame_files),
            "total_detections": 0,
            "max_tracks": 0,
            "processing_time": 0.0,
        }
        
        # 评估数据收集
        eval_predictions = {}
        eval_confidences = {}
        
        print(f"\n{'='*60}")
        print(f"传统CV方法跟踪: {image_dir.name}")
        print(f"总帧数: {len(frame_files)}, 图像尺寸: {w}x{h}")
        print(f"{'='*60}")
        
        for frame_idx, frame_path in enumerate(frame_files):
            t_start = time.time()
            
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # === 步骤1: HOG+SVM 行人检测 ===
            detections, confidences = self.detector.detect(frame)
            stats["total_detections"] += len(detections)
            
            # === 步骤2: 光流计算 ===
            if self.config.use_optical_flow:
                flow = self.flow_estimator.compute(gray)
            else:
                flow = None

            # === 步骤2.5: 光流校正 — 用光流位移修正卡尔曼预测 ===
            flow_corrections = None
            if flow is not None and self.config.flow_correction_weight > 0:
                flow_corrections = {}
                for track in self.tracker.tracks:
                    if not track.is_deleted:
                        bbox = track.get_bbox()
                        dx, dy = self.flow_estimator.get_bbox_motion(flow, bbox)
                        flow_corrections[track.track_id] = (dx, dy)

            # === 步骤3: 卡尔曼滤波跟踪 ===
            if len(detections) > 0:
                confirmed_tracks = self.tracker.update(detections, confidences=confidences, flow_corrections=flow_corrections)
            else:
                # 空检测时仍然更新（预测状态并衰减轨迹）
                empty_dets = np.zeros((0, 4), dtype=np.float32)
                empty_confs = np.zeros((0,), dtype=np.float32)
                confirmed_tracks = self.tracker.update(empty_dets, confidences=empty_confs, flow_corrections=flow_corrections)
            
            stats["max_tracks"] = max(stats["max_tracks"], len(confirmed_tracks))
            
            # 评估数据收集
            for track in confirmed_tracks:
                tid = track.track_id
                bbox = track.get_bbox()
                eval_predictions.setdefault(frame_idx, {})[tid] = bbox.copy()
                eval_confidences.setdefault(frame_idx, {})[tid] = getattr(track, 'confidence', 0.5)
            
            # === 步骤4: ReID 特征提取与注册 ===
            if self.config.use_reid and len(confirmed_tracks) > 0:
                for track in confirmed_tracks:
                    if track.time_since_update == 0:  # 刚匹配到的track
                        bbox = track.get_bbox()
                        features = self.feature_extractor.extract_features(frame, bbox)
                        self.reid_matcher.register(track.track_id, features)
            
            # === 步骤5: 可视化 ===
            vis_image = frame.copy()
            
            # 绘制检测结果
            for det, conf in zip(detections, confidences):
                x1, y1, x2, y2 = map(int, det)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (100, 100, 100), 1)
                cv2.putText(vis_image, f"{conf:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            # 绘制跟踪结果
            visualizer = TrackingVisualizer()
            for track in confirmed_tracks:
                tid = track.track_id
                bbox = track.get_bbox()
                color = visualizer.get_color(tid)
                
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID:{tid} H:{track.hits}"
                cv2.putText(vis_image, label, (x1, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 绘制轨迹
                if hasattr(track, 'history') and len(track.history) > 1:
                    history = track.history[-30:]
                    for i in range(1, len(history)):
                        cx1 = int((history[i-1][0] + history[i-1][2]) / 2)
                        cy1 = int((history[i-1][1] + history[i-1][3]) / 2)
                        cx2 = int((history[i][0] + history[i][2]) / 2)
                        cy2 = int((history[i][1] + history[i][3]) / 2)
                        alpha = i / len(history)
                        thickness = max(1, int(2 * alpha))
                        cv2.line(vis_image, (cx1, cy1), (cx2, cy2), color, thickness)
            
            # 帧信息
            cv2.putText(vis_image, f"Frame: {frame_idx+1}/{len(frame_files)} | "
                       f"Tracks: {len(confirmed_tracks)} | Detections: {len(detections)}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 方法标识
            cv2.putText(vis_image, "Traditional CV Method", (10, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 写入视频
            if video_writer:
                video_writer.write(vis_image)
            
            # 保存关键帧
            if self.config.save_frames and frame_idx % self.config.frame_skip == 0:
                cv2.imwrite(str(output_dir / "frames" / f"frame_{frame_idx:06d}.jpg"), vis_image)
            
            stats["processing_time"] += time.time() - t_start
            
            if verbose and (frame_idx + 1) % 20 == 0:
                fps = (frame_idx + 1) / stats["processing_time"]
                print(f"  帧 {frame_idx+1}/{len(frame_files)} | "
                      f"检测: {len(detections)} | 跟踪: {len(confirmed_tracks)} | "
                      f"FPS: {fps:.1f}")
        
        if video_writer:
            video_writer.release()
        
        # 保存MOT格式预测结果，供外部评估使用
        if eval_predictions:
            from tools.evaluate import TrackingEvaluator as _Eval
            _eval = _Eval()
            _eval.save_predictions_mot(eval_predictions, str(output_dir / "predictions.txt"), eval_confidences)
        
        avg_fps = len(frame_files) / stats["processing_time"]
        print(f"\n完成! 平均 FPS: {avg_fps:.1f}")
        print(f"最大同时跟踪目标: {stats['max_tracks']}")
        print(f"结果保存在: {output_dir}")
        
        return stats
    
    def process_video(
        self,
        video_path: Path,
        output_dir: Path,
        max_frames: Optional[int] = None,
        verbose: bool = True,
        eval_gt: bool = False,
        labels_dir: Optional[Path] = None,
        scene_name: Optional[str] = None,
    ) -> Dict:
        """
        处理视频文件

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            max_frames: 最大处理帧数
            verbose: 是否打印详细信息
            eval_gt: 是否启用GT评估
            labels_dir: YOLO格式GT标注目录
            scene_name: 场景名称过滤

        Returns:
            统计信息字典（包含评估指标如果eval_gt=True）
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

        # 统计信息
        stats = {
            "total_frames": 0,
            "total_detections": 0,
            "max_tracks": 0,
            "processing_time": 0.0,
        }

        # 评估数据收集
        eval_predictions = {}
        eval_confidences = {}

        print(f"\n{'='*60}")
        print(f"传统CV方法跟踪: {video_path.name}")
        print(f"视频FPS: {fps}, 总帧数: {total_frames}, 尺寸: {width}x{height}")
        print(f"处理帧数: {max_frames}")
        print(f"{'='*60}")

        visualizer = TrackingVisualizer()

        # 使用 tqdm 进度条
        pbar = tqdm(range(max_frames), desc="处理进度", unit="帧", ncols=100)
        for frame_idx in pbar:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # HOG+SVM 行人检测
            detections, confidences = self.detector.detect(frame)
            stats["total_detections"] += len(detections)

            # 光流计算
            if self.config.use_optical_flow:
                flow = self.flow_estimator.compute(gray)
            else:
                flow = None

            # 光流校正 — 用光流位移修正卡尔曼预测
            flow_corrections = None
            if flow is not None and self.config.flow_correction_weight > 0:
                flow_corrections = {}
                for track in self.tracker.tracks:
                    if not track.is_deleted:
                        bbox = track.get_bbox()
                        dx, dy = self.flow_estimator.get_bbox_motion(flow, bbox)
                        flow_corrections[track.track_id] = (dx, dy)

            # 卡尔曼滤波跟踪
            if len(detections) > 0:
                confirmed_tracks = self.tracker.update(detections, confidences=confidences, flow_corrections=flow_corrections)
            else:
                empty_dets = np.zeros((0, 4), dtype=np.float32)
                empty_confs = np.zeros((0,), dtype=np.float32)
                confirmed_tracks = self.tracker.update(empty_dets, confidences=empty_confs, flow_corrections=flow_corrections)

            stats["max_tracks"] = max(stats["max_tracks"], len(confirmed_tracks))

            # 评估数据收集（始终收集，供外部评估使用）
            for track in confirmed_tracks:
                tid = track.track_id
                bbox = track.get_bbox()
                eval_predictions.setdefault(frame_idx, {})[tid] = bbox.copy()
                eval_confidences.setdefault(frame_idx, {})[tid] = getattr(track, 'confidence', 0.5)

            # ReID 特征提取与注册
            if self.config.use_reid and len(confirmed_tracks) > 0:
                for track in confirmed_tracks:
                    if track.time_since_update == 0:
                        bbox = track.get_bbox()
                        features = self.feature_extractor.extract_features(frame, bbox)
                        self.reid_matcher.register(track.track_id, features)

            # 可视化
            vis_image = frame.copy()

            for det, conf in zip(detections, confidences):
                x1, y1, x2, y2 = map(int, det)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (100, 100, 100), 1)
                cv2.putText(vis_image, f"{conf:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            for track in confirmed_tracks:
                tid = track.track_id
                bbox = track.get_bbox()
                color = visualizer.get_color(tid)

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{tid} H:{track.hits}"
                cv2.putText(vis_image, label, (x1, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if hasattr(track, 'history') and len(track.history) > 1:
                    history = track.history[-30:]
                    for i in range(1, len(history)):
                        cx1 = int((history[i-1][0] + history[i-1][2]) / 2)
                        cy1 = int((history[i-1][1] + history[i-1][3]) / 2)
                        cx2 = int((history[i][0] + history[i][2]) / 2)
                        cy2 = int((history[i][1] + history[i][3]) / 2)
                        alpha = i / len(history)
                        thickness = max(1, int(2 * alpha))
                        cv2.line(vis_image, (cx1, cy1), (cx2, cy2), color, thickness)

            cv2.putText(vis_image, f"Frame: {frame_idx+1}/{max_frames} | "
                       f"Tracks: {len(confirmed_tracks)} | Detections: {len(detections)}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(vis_image, "Traditional CV Method", (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if video_writer:
                video_writer.write(vis_image)

            if self.config.save_frames and frame_idx % self.config.frame_skip == 0:
                cv2.imwrite(str(output_dir / "frames" / f"frame_{frame_idx:06d}.jpg"), vis_image)

            stats["total_frames"] += 1
            stats["processing_time"] += time.time() - t_start

            # 更新进度条信息
            pbar.set_postfix({
                "检测": len(detections),
                "跟踪": len(confirmed_tracks),
                "FPS": f"{(frame_idx + 1) / max(0.001, stats['processing_time']):.1f}"
            })

        pbar.close()
        cap.release()
        if video_writer:
            video_writer.release()

        # GT评估
        if eval_gt and labels_dir and labels_dir.exists():
            from tools.evaluate import TrackingEvaluator, TrackingMetrics

            print(f"\n{'='*60}")
            print("  GT评估")
            print(f"{'='*60}")

            evaluator = TrackingEvaluator(iou_threshold=0.5)
            gt = evaluator.load_yolo_gt(
                labels_dir,
                image_size=(width, height),
                scene_name=scene_name,
                class_ids=[0],
            )

            if gt:
                print(f"GT加载: {len(gt)}帧, 共{sum(len(v) for v in gt.values())}个标注")
                metrics = evaluator.evaluate(eval_predictions, gt)
                metrics.print_report()

                # 保存报告
                metrics.save_json(str(output_dir / "evaluation_report.json"))
                metrics.save_csv(str(output_dir / "evaluation_report.csv"))
                evaluator.save_predictions_mot(eval_predictions, str(output_dir / "predictions.txt"), eval_confidences)
                evaluator.save_gt_mot(gt, str(output_dir / "gt.txt"))

                # 将评估指标添加到 stats
                stats["MOTA"] = metrics.MOTA
                stats["MOTP"] = metrics.MOTP
                stats["IDF1"] = metrics.IDF1
                stats["IDSW"] = metrics.IDSW
                stats["FP"] = metrics.FP
                stats["FN"] = metrics.FN
                stats["TP"] = metrics.TP
                stats["Precision"] = metrics.precision
                stats["Recall"] = metrics.recall
                stats["MT"] = metrics.MT
                stats["ML"] = metrics.ML
                stats["frag"] = metrics.Frag
            else:
                print(f"警告: 未找到GT标注 (scene={scene_name})")
        elif eval_gt:
            print("\n[警告] 启用了评估但未提供GT标注目录 (--labels)")

        # 无条件保存MOT格式预测结果，供外部评估使用
        if eval_predictions:
            from tools.evaluate import TrackingEvaluator as _Eval
            _eval = _Eval()
            _eval.save_predictions_mot(eval_predictions, str(output_dir / "predictions.txt"), eval_confidences)

        avg_fps = stats["total_frames"] / max(stats["processing_time"], 0.001)
        print(f"\n完成! 平均 FPS: {avg_fps:.1f}")
        print(f"最大同时跟踪目标: {stats['max_tracks']}")
        print(f"结果保存在: {output_dir}")

        return stats

    def reset(self):
        """重置所有状态"""
        self.tracker.reset()
        self.flow_estimator.reset()
        if self.reid_matcher:
            self.reid_matcher.clear()
