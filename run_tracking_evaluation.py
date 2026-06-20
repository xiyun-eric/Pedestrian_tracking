"""
行人跟踪与评估脚本

运行 AdvancedTracker + standard预设 方案进行行人跟踪，
支持 GT 评估（Phase 4 评估流程集成）。

运行:
  python run_tracking_evaluation.py
  python run_tracking_evaluation.py --scene scene2
  python run_tracking_evaluation.py --eval --labels data/yolo_custom/labels/val --scene scene2
"""

import sys
import os
from pathlib import Path

# 禁用TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import time
import warnings

# 禁用警告
warnings.filterwarnings('ignore', category=UserWarning)

from deep_method.tracking import create_tracker, ReIDExtractor
from deep_method.detector import YOLODetector
from typing import Optional


def run_tracking(video_path: Path, output_dir: Path, max_frames: int = 200, use_custom_model: bool = True,
                 eval_gt: bool = False, labels_dir: Optional[Path] = None, scene_name: Optional[str] = None):
    """
    运行行人跟踪与评估

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        max_frames: 最大处理帧数
        use_custom_model: 是否使用自行训练的模型（True=本地训练，False=预训练yolo11m）
        eval_gt: 是否启用GT评估
        labels_dir: YOLO格式GT标注目录
        scene_name: 场景名称过滤
    """
    preset = 'standard'  # 固定使用standard预设
    print("="*60)
    print("行人跟踪与评估")
    print("="*60)
    print(f"视频: {video_path}")
    print(f"方案: AdvancedTracker + {preset}预设 + ReID")
    print(f"检测器: {'本地训练模型' if use_custom_model else '预训练yolo11m'}")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频FPS: {fps}, 总帧数: {total_frames}, 尺寸: {width}x{height}")
    print(f"处理帧数: {max_frames}")

    # 创建检测器（降低置信度阈值以启用ByteTrack低分框匹配）
    if use_custom_model:
        # 使用自行训练的模型（YOLO LoRA微调）
        model_path = project_root / "runs/yolo_lora/train/weights/best.pt"
        if model_path.exists():
            print(f"[模型] 加载本地训练模型: {model_path.resolve()}")
            print(f"[模型] 文件大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
            print(f"[模型] 确认加载微调权重 (非预训练)")
            detector = YOLODetector(
                model_path=str(model_path.resolve()),
                device='cpu',
                conf_threshold=0.1,  # 降低阈值，让ByteTrack处理低分框
                iou_threshold=0.5,   # NMS IoU阈值
                classes=[0],
            )
        else:
            print(f"[警告] 本地训练模型不存在: {model_path.resolve()}")
            print(f"[警告] 自动回退到预训练模型: {(project_root / 'yolo11m.pt').resolve()}")
            detector = YOLODetector(
                model_path=str(project_root / 'yolo11m.pt'),
                device='cpu',
                conf_threshold=0.1,
                iou_threshold=0.5,
                classes=[0],
            )
    else:
        # 使用预训练模型
        print(f"[模型] 加载预训练模型: {(project_root / 'yolo11m.pt').resolve()}")
        detector = YOLODetector(
            model_path=str(project_root / 'yolo11m.pt'),
            device='cpu',
            conf_threshold=0.1,
            iou_threshold=0.5,
            classes=[0],
        )

    # 创建跟踪器（使用指定预设）
    tracker = create_tracker(
        tracker_type='advanced',
        preset=preset,  # 使用指定的预设
        use_reid=True,  # 启用ReID
        reid_model='osnet_x1_0',  # 使用OSNet x1_0（MSMT17预训练权重，精度更高）
        device='cpu',
        use_torchreid=True,  # 使用torchreid加载OSNet
        use_finetuned_reid=use_custom_model,  # custom模式下优先使用微调ReID权重
    )

    # 从tracker获取ReID提取器（create_tracker内部已创建）
    reid = tracker.reid_extractor if hasattr(tracker, 'reid_extractor') else None

    if reid is not None:
        reid_tag = '微调' if use_custom_model else '预训练'
        print(f"[ReID] 使用OSNet x1_0（{reid_tag}权重）")
    else:
        print("[ReID] 已禁用，使用纯IoU+马氏距离匹配模式")

    # 视频写入器
    video_output = output_dir / "tracking_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_output), fourcc, int(fps), (width, height))

    # 统计
    stats = {
        'total_frames': 0,
        'total_detections': 0,
        'max_tracks': 0,
        'processing_time': 0,
    }

    # 评估数据收集
    eval_predictions = {}  # {frame_id: {track_id: bbox}}
    eval_confidences = {}  # {frame_id: {track_id: confidence}}

    # 颜色映射
    colors = {}

    # 处理帧
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()

        # 使用YOLO检测器进行检测
        detections, confidences, class_ids = detector.detect(frame)

        # ByteTrack: 分离高分框和低分框
        high_conf_threshold = tracker.config.high_conf_threshold  # 0.5
        if len(detections) > 0:
            high_mask = confidences >= high_conf_threshold
            high_dets = detections[high_mask]
            high_confs = confidences[high_mask]
            low_dets = detections[~high_mask]
            low_confs = confidences[~high_mask]
        else:
            high_dets = detections
            high_confs = confidences
            low_dets = np.zeros((0, 4), dtype=np.float32)
            low_confs = np.zeros((0,), dtype=np.float32)

        # ReID特征提取（仅高分框）
        features = None
        if reid is not None and len(high_dets) > 0:
            features = reid.extract_features_batch(frame, high_dets)

        # 跟踪更新（传递低分框给ByteTrack二次匹配）
        tracks = tracker.update(
            high_dets, high_confs,
            features=features, image=frame,
            low_conf_detections=low_dets if len(low_dets) > 0 else None,
            low_conf_confidences=low_confs if len(low_confs) > 0 else None,
        )

        # 统计轨迹状态
        active_tracks = [t for t in tracks if t.is_confirmed and t.time_since_update == 0]
        lost_tracks = [t for t in tracks if t.time_since_update > 0]

        stats['total_frames'] += 1
        stats['total_detections'] += len(detections)
        stats['max_tracks'] = max(stats['max_tracks'], len(tracks))
        stats['processing_time'] += time.time() - t_start

        # 可视化
        vis_frame = frame.copy()

        # 只绘制高分检测框（灰色，半透明）- 不绘制低分框
        for det, conf in zip(high_dets, high_confs):
            x1, y1, x2, y2 = map(int, det)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        # 只绘制活跃轨迹（time_since_update == 0），不绘制丢失轨迹
        for track in active_tracks:
            tid = track.track_id

            if tid not in colors:
                np.random.seed(tid)
                colors[tid] = tuple(np.random.randint(50, 255, 3).tolist())

            color = colors[tid]
            bbox = track.get_bbox()
            x1, y1, x2, y2 = map(int, bbox)

            # 收集评估数据
            if eval_gt:
                eval_predictions.setdefault(frame_count, {})[tid] = bbox.copy()
                eval_confidences.setdefault(frame_count, {})[tid] = track.confidence

            # 只绘制活跃轨迹
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid}"
            cv2.putText(vis_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 绘制轨迹
            if hasattr(track, 'history') and len(track.history) > 1:
                history = track.history[-30:]
                for i in range(1, len(history)):
                    cx1 = int((history[i-1][0] + history[i-1][2]) / 2)
                    cy1 = int((history[i-1][1] + history[i-1][3]) / 2)
                    cx2 = int((history[i][0] + history[i][2]) / 2)
                    cy2 = int((history[i][1] + history[i][3]) / 2)
                    alpha = i / len(history)
                    cv2.line(vis_frame, (cx1, cy1), (cx2, cy2), color, max(1, int(2*alpha)))

        # 帧信息
        cv2.putText(vis_frame,
                   f"Frame: {frame_count+1} | Tracks: {len(tracks)} | Active: {len(active_tracks)} | High: {len(high_dets)} | Low: {len(low_dets)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(vis_frame,
                   f"AdvancedTracker + {preset} + ByteTrack",
                   (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        video_writer.write(vis_frame)

        frame_count += 1
        if frame_count % 20 == 0:
            avg_fps = frame_count / stats['processing_time']
            print(f"帧 {frame_count}/{max_frames} | 检测: {len(detections)} | 跟踪: {len(tracks)} | FPS: {avg_fps:.1f}")

    cap.release()
    video_writer.release()

    # 打印统计
    avg_fps = stats['total_frames'] / stats['processing_time']
    print(f"\n结果:")
    print(f"  总帧数: {stats['total_frames']}")
    print(f"  总检测: {stats['total_detections']}")
    print(f"  最大跟踪数: {stats['max_tracks']}")
    print(f"  平均 FPS: {avg_fps:.1f}")
    print(f"  输出视频: {video_output}")

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
        else:
            print(f"警告: 未找到GT标注 (scene={scene_name})")
    elif eval_gt:
        print("\n[警告] 启用了评估但未提供GT标注目录 (--labels)")

    print("\n" + "="*60)
    print("跟踪与评估完成!")
    print("="*60)

    return {
        "FPS": stats['total_frames'] / stats['processing_time'],
    }


def run_tracking_images(image_dir: Path, output_dir: Path, max_frames: int = 200, use_custom_model: bool = True,
                        eval_gt: bool = False, labels_dir: Optional[Path] = None, scene_name: Optional[str] = None):
    """
    运行行人跟踪与评估（图像序列模式，如MOT17数据集）

    Args:
        image_dir: 图像序列目录路径
        output_dir: 输出目录
        max_frames: 最大处理帧数
        use_custom_model: 是否使用自行训练的模型（True=本地训练，False=预训练yolo11m）
        eval_gt: 是否启用GT评估
        labels_dir: YOLO格式GT标注目录
        scene_name: 场景名称过滤
    """
    preset = 'standard'  # 固定使用standard预设
    print("="*60)
    print("行人跟踪与评估（图像序列模式）")
    print("="*60)
    print(f"图像目录: {image_dir}")
    print(f"方案: AdvancedTracker + {preset}预设 + ReID")
    print(f"检测器: {'本地训练模型' if use_custom_model else '预训练yolo11m'}")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取排序后的图像文件列表
    frame_files = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    if not frame_files:
        print(f"错误: 图像目录中没有找到图像文件 {image_dir}")
        return

    total_frames = len(frame_files)
    # 读取第一帧获取尺寸
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"错误: 无法读取第一帧 {frame_files[0]}")
        return
    height, width = first_frame.shape[:2]
    fps = 10  # 图像序列默认fps

    if max_frames <= 0:
        max_frames = total_frames

    print(f"总帧数: {total_frames}, 尺寸: {width}x{height}, FPS: {fps}")
    print(f"处理帧数: {min(max_frames, total_frames)}")

    # 创建检测器（降低置信度阈值以启用ByteTrack低分框匹配）
    if use_custom_model:
        # 使用自行训练的模型（YOLO LoRA微调）
        model_path = project_root / "runs/yolo_lora/train/weights/best.pt"
        if model_path.exists():
            print(f"[模型] 加载本地训练模型: {model_path.resolve()}")
            print(f"[模型] 文件大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
            print(f"[模型] 确认加载微调权重 (非预训练)")
            detector = YOLODetector(
                model_path=str(model_path.resolve()),
                device='cpu',
                conf_threshold=0.1,
                iou_threshold=0.5,
                classes=[0],
            )
        else:
            print(f"[警告] 本地训练模型不存在: {model_path.resolve()}")
            print(f"[警告] 自动回退到预训练模型: {(project_root / 'yolo11m.pt').resolve()}")
            detector = YOLODetector(
                model_path=str(project_root / 'yolo11m.pt'),
                device='cpu',
                conf_threshold=0.1,
                iou_threshold=0.5,
                classes=[0],
            )
    else:
        # 使用预训练模型
        print(f"[模型] 加载预训练模型: {(project_root / 'yolo11m.pt').resolve()}")
        detector = YOLODetector(
            model_path=str(project_root / 'yolo11m.pt'),
            device='cpu',
            conf_threshold=0.1,
            iou_threshold=0.5,
            classes=[0],
        )

    # 创建跟踪器（使用指定预设）
    tracker = create_tracker(
        tracker_type='advanced',
        preset=preset,
        use_reid=True,
        reid_model='osnet_x1_0',
        device='cpu',
        use_torchreid=True,
        use_finetuned_reid=use_custom_model,
    )

    # 从tracker获取ReID提取器（create_tracker内部已创建）
    reid = tracker.reid_extractor if hasattr(tracker, 'reid_extractor') else None

    if reid is not None:
        reid_tag = '微调' if use_custom_model else '预训练'
        print(f"[ReID] 使用OSNet x1_0（{reid_tag}权重）")
    else:
        print("[ReID] 已禁用，使用纯IoU+马氏距离匹配模式")

    # 视频写入器
    video_output = output_dir / "tracking_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_output), fourcc, int(fps), (width, height))

    # 统计
    stats = {
        'total_frames': 0,
        'total_detections': 0,
        'max_tracks': 0,
        'processing_time': 0,
    }

    # 评估数据收集
    eval_predictions = {}  # {frame_id: {track_id: bbox}}
    eval_confidences = {}  # {frame_id: {track_id: confidence}}

    # 颜色映射
    colors = {}

    # 处理帧
    frame_count = 0
    frames_to_process = min(max_frames, total_frames)

    while frame_count < frames_to_process:
        frame = cv2.imread(str(frame_files[frame_count]))
        if frame is None:
            print(f"[警告] 无法读取帧: {frame_files[frame_count]}，跳过")
            frame_count += 1
            continue

        t_start = time.time()

        # 使用YOLO检测器进行检测
        detections, confidences, class_ids = detector.detect(frame)

        # ByteTrack: 分离高分框和低分框
        high_conf_threshold = tracker.config.high_conf_threshold  # 0.5
        if len(detections) > 0:
            high_mask = confidences >= high_conf_threshold
            high_dets = detections[high_mask]
            high_confs = confidences[high_mask]
            low_dets = detections[~high_mask]
            low_confs = confidences[~high_mask]
        else:
            high_dets = detections
            high_confs = confidences
            low_dets = np.zeros((0, 4), dtype=np.float32)
            low_confs = np.zeros((0,), dtype=np.float32)

        # ReID特征提取（仅高分框）
        features = None
        if reid is not None and len(high_dets) > 0:
            features = reid.extract_features_batch(frame, high_dets)

        # 跟踪更新（传递低分框给ByteTrack二次匹配）
        tracks = tracker.update(
            high_dets, high_confs,
            features=features, image=frame,
            low_conf_detections=low_dets if len(low_dets) > 0 else None,
            low_conf_confidences=low_confs if len(low_confs) > 0 else None,
        )

        # 统计轨迹状态
        active_tracks = [t for t in tracks if t.is_confirmed and t.time_since_update == 0]

        stats['total_frames'] += 1
        stats['total_detections'] += len(detections)
        stats['max_tracks'] = max(stats['max_tracks'], len(tracks))
        stats['processing_time'] += time.time() - t_start

        # 可视化
        vis_frame = frame.copy()

        # 只绘制高分检测框（灰色，半透明）- 不绘制低分框
        for det, conf in zip(high_dets, high_confs):
            x1, y1, x2, y2 = map(int, det)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

        # 只绘制活跃轨迹（time_since_update == 0），不绘制丢失轨迹
        for track in active_tracks:
            tid = track.track_id

            if tid not in colors:
                np.random.seed(tid)
                colors[tid] = tuple(np.random.randint(50, 255, 3).tolist())

            color = colors[tid]
            bbox = track.get_bbox()
            x1, y1, x2, y2 = map(int, bbox)

            # 收集评估数据（始终收集，供外部评估使用）
            eval_predictions.setdefault(frame_count, {})[tid] = bbox.copy()
            eval_confidences.setdefault(frame_count, {})[tid] = track.confidence

            # 只绘制活跃轨迹
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid}"
            cv2.putText(vis_frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 绘制轨迹
            if hasattr(track, 'history') and len(track.history) > 1:
                history = track.history[-30:]
                for i in range(1, len(history)):
                    cx1 = int((history[i-1][0] + history[i-1][2]) / 2)
                    cy1 = int((history[i-1][1] + history[i-1][3]) / 2)
                    cx2 = int((history[i][0] + history[i][2]) / 2)
                    cy2 = int((history[i][1] + history[i][3]) / 2)
                    alpha = i / len(history)
                    cv2.line(vis_frame, (cx1, cy1), (cx2, cy2), color, max(1, int(2*alpha)))

        # 帧信息
        cv2.putText(vis_frame,
                   f"Frame: {frame_count+1} | Tracks: {len(tracks)} | Active: {len(active_tracks)} | High: {len(high_dets)} | Low: {len(low_dets)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(vis_frame,
                   f"AdvancedTracker + {preset} + ByteTrack",
                   (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        video_writer.write(vis_frame)

        frame_count += 1
        if frame_count % 20 == 0:
            avg_fps = frame_count / stats['processing_time']
            print(f"帧 {frame_count}/{frames_to_process} | 检测: {len(detections)} | 跟踪: {len(tracks)} | FPS: {avg_fps:.1f}")

    video_writer.release()

    # 打印统计
    avg_fps = stats['total_frames'] / stats['processing_time']
    print(f"\n结果:")
    print(f"  总帧数: {stats['total_frames']}")
    print(f"  总检测: {stats['total_detections']}")
    print(f"  最大跟踪数: {stats['max_tracks']}")
    print(f"  平均 FPS: {avg_fps:.1f}")
    print(f"  输出视频: {video_output}")

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
        else:
            print(f"警告: 未找到GT标注 (scene={scene_name})")
    elif eval_gt:
        print("\n[警告] 启用了评估但未提供GT标注目录 (--labels)")

    # 无条件保存MOT格式预测结果，供外部评估使用
    if eval_predictions:
        from tools.evaluate import TrackingEvaluator as _Eval
        _eval = _Eval()
        _eval.save_predictions_mot(eval_predictions, str(output_dir / "predictions.txt"), eval_confidences)

    print("\n" + "="*60)
    print("跟踪与评估完成!")
    print("="*60)


def main():
    """主函数"""
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='行人跟踪与评估')
    parser.add_argument('--video', type=str, default='data/custom/videos/scene1.mp4',
                        help='视频文件路径')
    parser.add_argument('--images', type=str, default=None,
                        help='图像序列目录路径（如 data/MOT17/MOT17-04-FRCNN/img1），与 --video 二选一')
    parser.add_argument('--frames', type=int, default=200,
                        help='处理帧数，若设置为0表示处理全部帧')
    parser.add_argument('--model', type=str, default='custom',
                        choices=['custom', 'pretrained'],
                        help='模型类型: custom=本地训练模型, pretrained=预训练yolo11m')
    parser.add_argument('--eval', action='store_true',
                        help='启用GT评估')
    parser.add_argument('--labels', type=str, default='data/yolo_custom/labels/val',
                        help='YOLO格式GT标注目录')
    parser.add_argument('--scene', type=str, default=None,
                        help='场景名称过滤（如scene1, scene2）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录（默认自动生成）')

    args = parser.parse_args()

    # 模型选择
    use_custom_model = (args.model == 'custom')

    if args.images:
        # 图像序列模式
        image_dir = Path(args.images)
        if not image_dir.exists():
            print(f"错误: 图像目录不存在 {image_dir}")
            return

        if args.output:
            output_dir = Path(args.output)
        else:
            # 输出目录结构: outputs/{model}/{scene_name}/
            output_dir = Path("outputs") / args.model / image_dir.name

        run_tracking_images(image_dir, output_dir, max_frames=args.frames,
                           use_custom_model=use_custom_model,
                           eval_gt=args.eval, labels_dir=Path(args.labels) if args.eval else None,
                           scene_name=args.scene)
    else:
        # 视频模式
        if args.scene and args.video == parser.get_default('video'):
            video_path = Path(f"data/custom/videos/{args.scene}.mp4")
        else:
            video_path = Path(args.video)

        if not video_path.exists():
            print(f"错误: 视频文件不存在 {video_path}")
            return

        if args.output:
            output_dir = Path(args.output)
        else:
            # 输出目录结构: outputs/{model}/{scene_name}/
            output_dir = Path("outputs") / args.model / video_path.stem

        # 运行跟踪与评估（固定使用standard预设）
        run_tracking(video_path, output_dir, max_frames=args.frames,
                     use_custom_model=use_custom_model,
                     eval_gt=args.eval, labels_dir=Path(args.labels) if args.eval else None,
                     scene_name=args.scene)


if __name__ == "__main__":
    main()
