"""
对比实验主控脚本

统一调度三种方法，收集结果，调用对比模块生成报告。

三种对比方法:
  (i)   传统方法 (HOG+SVM + Kalman)        - traditional
  (ii)  微调后的深度方法 (YOLO11+LoRA)      - deep_custom
  (iii) 大模型方法 (Qwen2-VL+LoRA+IoU跟踪) - large_model

使用方法:
  # MOT17 模式: 对 02, 04, 11 三个场景运行三种方法，只评估前 10 帧
  python run_comparison.py --mot17
  
  # 自定义场景和帧数
  python run_comparison.py --mot17 --mot17-seq MOT17-02-FRCNN MOT17-04-FRCNN MOT17-11-FRCNN --mot17-frames 10
  
  # 只运行特定方法
  python run_comparison.py --mot17 --methods traditional large_model
  
  # 自收集视频模式 (原有逻辑)
  python run_comparison.py --videos-scenes scene1 scene2 scene3 scene4 scene5 --frames 150
"""

import sys
import os
from pathlib import Path
import subprocess
import json
import argparse
import re
from typing import List, Dict, Optional
from tqdm import tqdm

import numpy as np

# 添加项目根目录到路径
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 固定参数
CUSTOM_VIDEOS_DIR = _project_root / "data" / "custom" / "videos"
CUSTOM_LABELS_DIR = _project_root / "data" / "yolo_custom" / "labels" / "val"
KITTI_DIR = _project_root / "data" / "kitti"
KITTI_LABELS_DIR = _project_root / "data" / "kitti" / "labels"
MOT17_DIR = _project_root / "data" / "MOT17"
DEFAULT_FRAMES = 150


def check_gt_available(scene_name: str) -> bool:
    """检查场景是否有GT标注"""
    label_dir = CUSTOM_LABELS_DIR
    if not label_dir.exists():
        return False
    # 检查是否有 sceneX_frame_*.txt 文件
    pattern = f"{scene_name}_frame_*.txt"
    matches = list(label_dir.glob(pattern))
    return len(matches) > 0


def check_kitti_gt_available(seq_name: str) -> bool:
    """检查KITTI序列是否有GT标注"""
    label_dir = KITTI_LABELS_DIR
    if not label_dir.exists():
        return False
    pattern = f"{seq_name}_*.txt"
    matches = list(label_dir.glob(pattern))
    return len(matches) > 0


def run_traditional_method(video_path: Path, scene_name: str, 
                           has_gt: bool, frames: int,
                           output_dir: Optional[Path] = None,
                           use_hog_api: bool = True,
                           use_svm_api: bool = True) -> Optional[Dict]:
    """
    运行传统方法 (HOG+SVM + Kalman)
    
    Args:
        video_path: 视频路径
        scene_name: 场景名称
        has_gt: 是否有GT标注
        frames: 处理帧数
        output_dir: 输出目录
        use_hog_api: HOG特征提取是否使用OpenCV API（默认True）
        use_svm_api: SVM是否使用OpenCV预训练权重（默认True）
    
    Returns:
        评估指标字典，包含MOTA/MOTP/FPS等（如果有GT），否则只包含FPS
    """
    if output_dir is None:
        output_dir = _project_root / "outputs" / "traditional" / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(_project_root / "traditional_method" / "run_traditional.py"),
        "--video", str(video_path),
        "--output-dir", str(output_dir),
        "--frames", str(frames),
    ]
    
    # 添加API参数
    if not use_hog_api:
        cmd.append("--no-hog-api")
    if not use_svm_api:
        cmd.append("--no-svm-api")
    
    if has_gt:
        cmd.extend([
            "--eval",
            "--labels", str(CUSTOM_LABELS_DIR),
            "--scene", scene_name,
        ])
    
    print(f"\n{'='*70}")
    print(f"  传统方法 (HOG+SVM): {scene_name} {'[有GT]' if has_gt else '[无GT-仅FPS]'}")
    print(f"  HOG模式: {'OpenCV API' if use_hog_api else '手动实现'}")
    print(f"  SVM模式: {'预训练权重' if use_svm_api else '手动实现'}")
    print(f"{'='*70}")
    
    # 实时显示子进程输出
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True, 
        encoding='utf-8', 
        errors='replace',
        cwd=_project_root
    )
    
    # 实时读取并打印输出
    output_lines = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line, end='')
            output_lines.append(line)
    
    process.wait(timeout=600)
    stdout = ''.join(output_lines)
    stderr = ""
    
    # 解析输出中的统计信息
    stats = _extract_stats_from_output(stdout, stderr)
    
    if process.returncode != 0:
        print(f"[警告] 传统方法运行有错误 (returncode={process.returncode})")
    
    if has_gt:
        eval_json = output_dir / "evaluation_report.json"
        if eval_json.exists():
            with open(eval_json, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            metrics["FPS"] = stats.get("fps", 0.0)
            metrics["method"] = "传统方法 (HOG+SVM)"
            metrics["scene"] = scene_name
            return metrics
        else:
            print(f"[警告] 未找到评估文件: {eval_json}")
    
    # 无GT时返回基础统计
    return {
        "method": "传统方法 (HOG+SVM)",
        "scene": scene_name,
        "FPS": stats.get("fps", 0.0),
        "total_frames": stats.get("total_frames", frames),
        "processing_time": stats.get("processing_time", 0),
        "MOTA": "N/A",
        "MOTP": "N/A",
        "IDF1": "N/A",
        "IDSW": "N/A",
        "FP": "N/A",
        "FN": "N/A",
        "TP": "N/A",
        "GT": "N/A",
        "Precision": "N/A",
        "Recall": "N/A",
        "MT": "N/A",
        "ML": "N/A",
        "frag": "N/A",
        "num_frames": "N/A",
    }


def run_deep_method(video_path: Path, scene_name: str,
                    has_gt: bool, frames: int, model_type: str) -> Optional[Dict]:
    """
    运行深度方法 (YOLO11 + AdvancedTracker)
    
    Args:
        model_type: 'pretrained' 或 'custom'
    """
    cmd = [
        sys.executable,
        str(_project_root / "run_tracking_evaluation.py"),
        "--video", str(video_path),
        "--model", model_type,
        "--frames", str(frames),
    ]
    
    if has_gt:
        cmd.extend([
            "--eval",
            "--labels", str(CUSTOM_LABELS_DIR),
            "--scene", scene_name,
        ])
    
    model_label = "微调" if model_type == 'custom' else "原始"
    method_name = f"深度方法 ({model_label})"
    
    print(f"\n{'='*70}")
    print(f"  {method_name}: {scene_name} {'[有GT]' if has_gt else '[无GT-仅FPS]'}")
    print(f"{'='*70}")
    
    # 实时显示子进程输出
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True, 
        encoding='utf-8', 
        errors='replace',
        cwd=_project_root
    )
    
    # 实时读取并打印输出
    output_lines = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line, end='')
            output_lines.append(line)
    
    process.wait(timeout=600)
    stdout = ''.join(output_lines)
    stderr = ""
    
    stats = _extract_stats_from_output(stdout, stderr)
    
    if process.returncode != 0:
        print(f"[警告] 深度方法运行有错误 (returncode={process.returncode})")
    
    if has_gt:
        # 读取评估文件（更新后的输出目录结构）
        expected_output_dir = _project_root / "outputs" / model_type / video_path.stem
        eval_json = expected_output_dir / "evaluation_report.json"
        if eval_json.exists():
            with open(eval_json, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            metrics["FPS"] = stats.get("fps", 0.0)
            metrics["method"] = method_name
            metrics["scene"] = scene_name
            return metrics
        else:
            print(f"[警告] 未找到评估文件: {eval_json}")
    
    return {
        "method": method_name,
        "scene": scene_name,
        "FPS": stats.get("fps", 0.0),
        "total_frames": stats.get("total_frames", frames),
        "processing_time": stats.get("processing_time", 0),
        "MOTA": "N/A",
        "MOTP": "N/A",
        "IDF1": "N/A",
        "IDSW": "N/A",
        "FP": "N/A",
        "FN": "N/A",
        "TP": "N/A",
        "GT": "N/A",
        "Precision": "N/A",
        "Recall": "N/A",
        "MT": "N/A",
        "ML": "N/A",
        "frag": "N/A",
        "num_frames": "N/A",
    }


def run_large_model_method(seq_names: List[str], frames: int,
                           output_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    运行大模型方法 (Qwen2-VL + LoRA + IoU 后处理跟踪)

    通过调用 model_method/visualize.py --mode detect_track 完成。
    visualize.py 会输出 evaluation_report.json，其中包含 'overall' 和 'per_sequence' 字段。

    Args:
        seq_names: MOT17 序列名称列表
        frames: 每个序列处理帧数
        output_dir: 大模型可视化输出目录

    Returns:
        {seq_name: metrics_dict} 每个序列的指标字典
    """
    if output_dir is None:
        output_dir = _project_root / "outputs" / "large_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    # LoRA 权重路径 (训练结果已迁移至 runs/)
    lora_path = _project_root / "runs" / "stage1" / "final"
    if not lora_path.exists():
        # 兼容旧路径 output/stage1/final
        legacy_lora = _project_root / "output" / "stage1" / "final"
        if legacy_lora.exists():
            lora_path = legacy_lora
        else:
            print(f"[警告] LoRA 权重不存在: {lora_path} (也无旧路径 {legacy_lora})")
            return {seq: {"method": "大模型方法 (Qwen2-VL+LoRA)", "scene": seq,
                          "MOTA": "N/A", "MOTP": "N/A", "IDF1": "N/A", "IDSW": "N/A",
                          "FPS": 0.0} for seq in seq_names}

    # 序列过滤参数
    seq_filter = ",".join(seq_names)

    cmd = [
        sys.executable,
        str(_project_root / "model_method" / "visualize.py"),
        "--mode", "detect_track",
        "--lora-path", str(lora_path),
        "--output-dir", str(output_dir),
        "--seq-filter", seq_filter,
        "--max-frames", str(frames),
    ]

    print(f"\n{'='*70}")
    print(f"  大模型方法 (Qwen2-VL+LoRA): 序列={seq_names} 帧数={frames}")
    print(f"{'='*70}")

    # 实时显示子进程输出
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=_project_root
    )

    output_lines = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line, end='')
            output_lines.append(line)

    process.wait(timeout=3600)  # 大模型推理较慢，给 1 小时
    stdout = ''.join(output_lines)
    stderr = ""

    stats = _extract_stats_from_output(stdout, stderr)

    if process.returncode != 0:
        print(f"[警告] 大模型方法运行有错误 (returncode={process.returncode})")

    # 读取 visualize.py 输出的 evaluation_report.json
    # visualize_detect_track 函数会在 output_dir 下创建 track_detect_iou 子目录
    report_path = output_dir / "track_detect_iou" / "evaluation_report.json"
    per_seq_metrics = {}
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        per_seq_data = report.get('per_sequence', {})
        for seq_name in seq_names:
            if seq_name in per_seq_data:
                m = per_seq_data[seq_name]
                m["FPS"] = stats.get("fps", 0.0)
                m["method"] = "大模型方法 (Qwen2-VL+LoRA)"
                m["scene"] = seq_name
                per_seq_metrics[seq_name] = m
            else:
                per_seq_metrics[seq_name] = {
                    "method": "大模型方法 (Qwen2-VL+LoRA)", "scene": seq_name,
                    "FPS": stats.get("fps", 0.0),
                    "MOTA": "N/A", "MOTP": "N/A", "IDF1": "N/A", "IDSW": "N/A",
                    "FP": "N/A", "FN": "N/A", "TP": "N/A", "GT": "N/A",
                    "Precision": "N/A", "Recall": "N/A",
                }
    else:
        print(f"[警告] 未找到大模型评估报告: {report_path}")
        for seq_name in seq_names:
            per_seq_metrics[seq_name] = {
                "method": "大模型方法 (Qwen2-VL+LoRA)", "scene": seq_name,
                "FPS": stats.get("fps", 0.0),
                "MOTA": "N/A", "MOTP": "N/A", "IDF1": "N/A", "IDSW": "N/A",
                "FP": "N/A", "FN": "N/A", "TP": "N/A", "GT": "N/A",
                "Precision": "N/A", "Recall": "N/A",
            }

    return per_seq_metrics


def _extract_stats_from_output(stdout: str, stderr: str) -> Dict:
    """从输出文本中提取FPS等统计信息"""
    stats = {}
    combined = (stdout or "") + "\n" + (stderr or "")

    # 尝试提取FPS
    fps_patterns = [
        r'FPS[:\s]*([\d.]+)',
        r'fps[:\s]*([\d.]+)',
        r'平均[Ff][Pp][Ss][:\s]*([\d.]+)',
        r'avg_fps[:\s]*([\d.]+)',
        r'processing_time[:\s]*([\d.]+)',
        r'total_frames[:\s]*(\d+)',
    ]

    for pattern in fps_patterns:
        match = re.search(pattern, combined)
        if match:
            key = pattern.split(':')[0].split('(')[0].strip().lower().replace('[', '').replace(']', '')
            value = float(match.group(1))
            if 'fps' in pattern.lower():
                stats['fps'] = value
            elif 'processing_time' in pattern.lower():
                stats['processing_time'] = value
            elif 'total_frames' in pattern.lower():
                stats['total_frames'] = int(value)

    # 如果没找到fps，尝试从 total_frames / processing_time 计算
    if 'fps' not in stats and 'total_frames' in stats and 'processing_time' in stats:
        if stats['processing_time'] > 0:
            stats['fps'] = stats['total_frames'] / stats['processing_time']

    return stats


def _load_mot_predictions(pred_path: Path) -> Dict:
    """
    加载MOT格式的预测结果文件

    MOT格式: <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, <class>, <visibility>
    转换为 {frame_id: {track_id: [x1, y1, x2, y2]}}

    Args:
        pred_path: predictions.txt 文件路径

    Returns:
        {frame_id: {track_id: [x1, y1, x2, y2]}}
    """
    from collections import defaultdict
    predictions = defaultdict(dict)

    if not pred_path.exists():
        return dict(predictions)

    with open(pred_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])  # MOT格式帧号保持原样（1-based），与GT一致
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            # 转换 (x, y, w, h) -> [x1, y1, x2, y2]
            predictions[frame_id][track_id] = np.array(
                [x, y, x + w, y + h], dtype=np.float32
            )

    return dict(predictions)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='对比实验主控脚本 - 自收集多目标跟踪数据集')
    parser.add_argument('--videos-scenes', type=str, nargs='+',
                        default=['scene1', 'scene2', 'scene3', 'scene4', 'scene5'],
                        help='要测试的场景名称列表 (scene1, scene2, ...)')
    parser.add_argument('--methods', type=str, nargs='+',
                        choices=['traditional', 'deep_pretrained', 'deep_custom', 'large_model', 'all'],
                        default=['all'],
                        help='要运行的方法')
    parser.add_argument('--frames', type=int, default=DEFAULT_FRAMES,
                        help='每个视频处理帧数')
    parser.add_argument('--output-dir', type=str, default='outputs/comparison',
                        help='对比报告输出目录')
    parser.add_argument('--kitti', action='store_true',
                        help='使用KITTI数据集模式')
    parser.add_argument('--kitti-dir', type=str, default='data/kitti',
                        help='KITTI数据集图像目录')
    parser.add_argument('--kitti-seq', type=str, nargs='+', default=['0017', '0019'],
                        help='KITTI序列名称')
    parser.add_argument('--mot17', action='store_true',
                        help='使用MOT17数据集模式')
    parser.add_argument('--mot17-dir', type=str, default='data/MOT17',
                        help='MOT17数据集根目录')
    parser.add_argument('--mot17-seq', type=str, nargs='+',
                        default=['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-11-FRCNN'],
                        help='MOT17序列名称（默认: 02, 04, 11 三个场景）')
    parser.add_argument('--mot17-frames', type=int, default=10,
                        help='MOT17模式每个序列处理帧数 (默认: 10，避免评估时间过长)')
    parser.add_argument('--large-model-output', type=str, default='outputs/large_model',
                        help='大模型方法可视化输出目录')
    parser.add_argument('--use-hog-api', action="store_true", default=True,
                        help="传统方法HOG特征提取使用OpenCV API（默认开启）")
    parser.add_argument('--no-hog-api', action="store_true",
                        help="传统方法HOG特征提取使用手动实现（非常慢）")
    parser.add_argument('--use-svm-api', action="store_true", default=True,
                        help="传统方法SVM使用OpenCV预训练权重（默认开启）")
    parser.add_argument('--no-svm-api', action="store_true",
                        help="传统方法SVM不使用OpenCV预训练权重")

    args = parser.parse_args()

    # 判断是否使用API
    use_hog_api = not args.no_hog_api
    use_svm_api = not args.no_svm_api

    # 确定要运行的方法
    if 'all' in args.methods:
        # 默认对比三种方法: 传统HOG、微调深度方法、大模型方案
        methods_to_run = ['traditional', 'deep_custom', 'large_model']
    else:
        methods_to_run = args.methods

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化对比报告生成器
    from tools.comparison import ComparisonReporter
    reporter = ComparisonReporter(output_dir)

    # KITTI数据集模式
    if args.kitti:
        kitti_dir = Path(args.kitti_dir)
        frames = args.frames

        for seq_name in args.kitti_seq:
            seq_dir = kitti_dir / seq_name
            if not seq_dir.exists():
                print(f"[警告] KITTI序列不存在: {seq_dir}，跳过")
                continue

            has_gt = check_kitti_gt_available(seq_name)
            if has_gt:
                print(f"\n[信息] KITTI {seq_name}: 检测到GT标注")
            else:
                print(f"\n[信息] KITTI {seq_name}: 无GT标注，仅收集FPS/处理时间")

            # 传统方法
            if 'traditional' in methods_to_run:
                cmd = [
                    sys.executable,
                    str(_project_root / "traditional_method" / "run_traditional.py"),
                    "--data-dir", str(kitti_dir), "--seq", seq_name,
                    "--frames", str(frames),
                ]
                if has_gt:
                    cmd.extend([
                        "--eval",
                        "--labels", str(KITTI_LABELS_DIR),
                        "--scene", seq_name,
                    ])

                print(f"\n{'='*70}")
                print(f"  传统方法 (HOG+SVM): KITTI-{seq_name} {'[有GT]' if has_gt else '[无GT-仅FPS]'}")
                print(f"{'='*70}")

                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=_project_root, timeout=600)
                stats = _extract_stats_from_output(result.stdout, result.stderr)

                if result.returncode != 0:
                    print(f"[警告] 传统方法运行有错误 (returncode={result.returncode})")
                    print(result.stderr[-1500:])

                trad_output_dir = _project_root / "outputs" / "traditional" / f"kitti_{seq_name}"
                if has_gt:
                    eval_json = trad_output_dir / "evaluation_report.json"
                    if eval_json.exists():
                        with open(eval_json, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        metrics["FPS"] = stats.get("fps", 0.0)
                        metrics["method"] = "传统方法 (HOG+SVM)"
                        metrics["scene"] = f"KITTI-{seq_name}"
                        reporter.add_result(metrics["method"], f"KITTI-{seq_name}", metrics)
                    else:
                        reporter.add_result("传统方法 (HOG+SVM)", f"KITTI-{seq_name}",
                            {"method": "传统方法 (HOG+SVM)", "scene": f"KITTI-{seq_name}",
                             "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A"})
                else:
                    reporter.add_result("传统方法 (HOG+SVM)", f"KITTI-{seq_name}",
                        {"method": "传统方法 (HOG+SVM)", "scene": f"KITTI-{seq_name}",
                         "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A"})

            # 深度方法 (预训练)
            if 'deep_pretrained' in methods_to_run:
                cmd = [
                    sys.executable,
                    str(_project_root / "run_tracking_evaluation.py"),
                    "--images", str(seq_dir),
                    "--model", "pretrained",
                    "--frames", str(frames),
                ]
                if has_gt:
                    cmd.extend([
                        "--eval",
                        "--labels", str(KITTI_LABELS_DIR),
                        "--scene", seq_name,
                    ])

                print(f"\n{'='*70}")
                print(f"  深度方法 (原始): KITTI-{seq_name} {'[有GT]' if has_gt else '[无GT-仅FPS]'}")
                print(f"{'='*70}")

                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=_project_root, timeout=600)
                stats = _extract_stats_from_output(result.stdout, result.stderr)

                if result.returncode != 0:
                    print(f"[警告] 深度方法运行有错误 (returncode={result.returncode})")
                    print(result.stderr[-1500:])

                expected_output_dir = _project_root / "tracking_evaluation" / seq_name / "pretrained"
                if has_gt:
                    eval_json = expected_output_dir / "evaluation_report.json"
                    if eval_json.exists():
                        with open(eval_json, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        metrics["FPS"] = stats.get("fps", 0.0)
                        metrics["method"] = "深度方法 (原始)"
                        metrics["scene"] = f"KITTI-{seq_name}"
                        reporter.add_result(metrics["method"], f"KITTI-{seq_name}", metrics)
                    else:
                        reporter.add_result("深度方法 (原始)", f"KITTI-{seq_name}",
                            {"method": "深度方法 (原始)", "scene": f"KITTI-{seq_name}",
                             "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A"})
                else:
                    reporter.add_result("深度方法 (原始)", f"KITTI-{seq_name}",
                        {"method": "深度方法 (原始)", "scene": f"KITTI-{seq_name}",
                         "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A"})

            # 深度方法 (微调)
            if 'deep_custom' in methods_to_run:
                cmd = [
                    sys.executable,
                    str(_project_root / "run_tracking_evaluation.py"),
                    "--images", str(seq_dir),
                    "--model", "custom",
                    "--frames", str(frames),
                ]
                if has_gt:
                    cmd.extend([
                        "--eval",
                        "--labels", str(KITTI_LABELS_DIR),
                        "--scene", seq_name,
                    ])

                print(f"\n{'='*70}")
                print(f"  深度方法 (微调): KITTI-{seq_name} {'[有GT]' if has_gt else '[无GT-仅FPS]'}")
                print(f"{'='*70}")

                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=_project_root, timeout=600)
                stats = _extract_stats_from_output(result.stdout, result.stderr)

                if result.returncode != 0:
                    print(f"[警告] 深度方法运行有错误 (returncode={result.returncode})")
                    print(result.stderr[-1500:])

                expected_output_dir = _project_root / "tracking_evaluation" / seq_name / "custom"
                if has_gt:
                    eval_json = expected_output_dir / "evaluation_report.json"
                    if eval_json.exists():
                        with open(eval_json, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                        metrics["FPS"] = stats.get("fps", 0.0)
                        metrics["method"] = "深度方法 (微调)"
                        metrics["scene"] = f"KITTI-{seq_name}"
                        reporter.add_result(metrics["method"], f"KITTI-{seq_name}", metrics)
                    else:
                        reporter.add_result("深度方法 (微调)", f"KITTI-{seq_name}",
                            {"method": "深度方法 (微调)", "scene": f"KITTI-{seq_name}",
                             "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A"})
                else:
                    reporter.add_result("深度方法 (微调)", f"KITTI-{seq_name}",
                        {"method": "深度方法 (微调)", "scene": f"KITTI-{seq_name}",
                         "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A"})

        # 生成对比报告（KITTI模式）
        print(f"\n{'='*70}")
        print("  生成对比报告")
        print(f"{'='*70}")

        reporter.save_json_report()
        csv_path = output_dir / "comparison_table.csv"
        reporter.generate_table_csv(csv_path)
        reporter.print_summary()

        markdown_table = reporter.generate_table_markdown()
        print(f"\n{'='*70}")
        print("  Markdown 对比表格")
        print(f"{'='*70}")
        print(markdown_table)

        md_path = output_dir / "comparison_table.md"
        md_path.write_text(markdown_table, encoding='utf-8')
        print(f"\nMarkdown表格已保存: {md_path}")

        print(f"\n所有报告保存至: {output_dir}/")
        for f in sorted(output_dir.glob("*")):
            print(f"  - {f.name}")

        print(f"\n{'='*70}")
        print("  KITTI对比实验完成!")
        print(f"{'='*70}")
        return

    # MOT17数据集模式
    if args.mot17:
        mot17_dir = Path(args.mot17_dir)
        # MOT17 模式使用 --mot17-frames (默认 10 帧，避免评估时间过长)
        frames = args.mot17_frames

        # 自动发现序列 (默认使用 02, 04, 11 三个场景)
        if args.mot17_seq:
            seq_names = args.mot17_seq
        else:
            seq_names = sorted([d.name for d in mot17_dir.iterdir()
                               if d.is_dir() and d.name.startswith('MOT17')])
            if not seq_names:
                print(f"[错误] 未在 {mot17_dir} 中找到MOT17序列")
                return
            print(f"[信息] 自动发现 {len(seq_names)} 个MOT17序列: {seq_names}")

        for seq_name in seq_names:
            seq_dir = mot17_dir / seq_name
            img_dir = seq_dir / "img1"
            gt_file = seq_dir / "gt" / "gt.txt"

            if not img_dir.exists():
                print(f"[警告] 图像目录不存在: {img_dir}，跳过")
                continue

            has_gt = gt_file.exists()
            if has_gt:
                print(f"\n[信息] MOT17 {seq_name}: 检测到GT标注")
            else:
                print(f"\n[信息] MOT17 {seq_name}: 无GT标注，仅收集FPS/处理时间")

            # 读取 seqinfo.ini 获取图像尺寸
            seqinfo_path = seq_dir / "seqinfo.ini"
            img_w, img_h = 1920, 1080  # 默认值
            if seqinfo_path.exists():
                with open(seqinfo_path, 'r') as f:
                    for line in f:
                        if line.startswith('imWidth'):
                            img_w = int(line.split('=')[1].strip())
                        elif line.startswith('imHeight'):
                            img_h = int(line.split('=')[1].strip())

            # 传统方法
            if 'traditional' in methods_to_run:
                trad_output_dir = _project_root / "outputs" / "traditional" / f"seq_{seq_name}"
                trad_output_dir.mkdir(parents=True, exist_ok=True)

                cmd = [
                    sys.executable,
                    str(_project_root / "traditional_method" / "run_traditional.py"),
                    "--data-dir", str(mot17_dir), "--seq", seq_name,
                    "--frames", str(frames),
                ]

                print(f"\n{'='*70}")
                print(f"  传统方法 (HOG+SVM): {seq_name} {'[有GT]' if has_gt else '[无GT-仅FPS]'}")
                print(f"{'='*70}")

                result = subprocess.run(cmd, capture_output=True, text=True,
                                       encoding='utf-8', errors='replace',
                                       cwd=_project_root, timeout=600)
                stats = _extract_stats_from_output(result.stdout, result.stderr)

                if result.returncode != 0:
                    print(f"[警告] 传统方法运行有错误 (returncode={result.returncode})")
                    print(result.stderr[-1500:])

                # MOT17 GT评估
                if has_gt:
                    from tools.evaluate import TrackingEvaluator
                    evaluator = TrackingEvaluator(iou_threshold=0.5)
                    gt = evaluator.load_mot_gt(gt_file, class_ids=[1])

                    pred_file = trad_output_dir / "predictions.txt"
                    if pred_file.exists():
                        predictions = _load_mot_predictions(pred_file)
                        metrics_obj = evaluator.evaluate(predictions, gt)
                        metrics = metrics_obj.to_dict()
                        metrics["FPS"] = stats.get("fps", 0.0)
                        metrics["method"] = "传统方法 (HOG+SVM)"
                        metrics["scene"] = seq_name
                        reporter.add_result(metrics["method"], seq_name, metrics)
                    else:
                        reporter.add_result("传统方法 (HOG+SVM)", seq_name,
                            {"method": "传统方法 (HOG+SVM)", "scene": seq_name,
                             "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A",
                             "IDF1": "N/A", "IDSW": "N/A", "Precision": "N/A", "Recall": "N/A"})
                else:
                    reporter.add_result("传统方法 (HOG+SVM)", seq_name,
                        {"method": "传统方法 (HOG+SVM)", "scene": seq_name,
                         "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A"})

            # 深度方法 (预训练)
            if 'deep_pretrained' in methods_to_run:
                deep_output_dir = _project_root / "tracking_evaluation" / seq_name / "pretrained"
                cmd = [
                    sys.executable,
                    str(_project_root / "run_tracking_evaluation.py"),
                    "--images", str(img_dir),
                    "--model", "pretrained",
                    "--frames", str(frames),
                    "--output", str(deep_output_dir),
                ]

                print(f"\n{'='*70}")
                print(f"  深度方法 (原始): {seq_name} {'[有GT]' if has_gt else '[无GT-仅FPS]'}")
                print(f"{'='*70}")

                result = subprocess.run(cmd, capture_output=True, text=True,
                                       encoding='utf-8', errors='replace',
                                       cwd=_project_root, timeout=600)
                stats = _extract_stats_from_output(result.stdout, result.stderr)

                if result.returncode != 0:
                    print(f"[警告] 深度方法运行有错误 (returncode={result.returncode})")
                    print(result.stderr[-1500:])

                if has_gt:
                    from tools.evaluate import TrackingEvaluator
                    evaluator = TrackingEvaluator(iou_threshold=0.5)
                    gt = evaluator.load_mot_gt(gt_file, class_ids=[1])

                    pred_file = deep_output_dir / "predictions.txt"
                    if pred_file.exists():
                        predictions = _load_mot_predictions(pred_file)
                        metrics_obj = evaluator.evaluate(predictions, gt)
                        metrics = metrics_obj.to_dict()
                        metrics["FPS"] = stats.get("fps", 0.0)
                        metrics["method"] = "深度方法 (原始)"
                        metrics["scene"] = seq_name
                        reporter.add_result(metrics["method"], seq_name, metrics)
                    else:
                        reporter.add_result("深度方法 (原始)", seq_name,
                            {"method": "深度方法 (原始)", "scene": seq_name,
                             "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A",
                             "IDF1": "N/A", "IDSW": "N/A", "Precision": "N/A", "Recall": "N/A"})
                else:
                    reporter.add_result("深度方法 (原始)", seq_name,
                        {"method": "深度方法 (原始)", "scene": seq_name,
                         "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A"})

            # 深度方法 (微调)
            if 'deep_custom' in methods_to_run:
                deep_output_dir = _project_root / "tracking_evaluation" / seq_name / "custom"
                cmd = [
                    sys.executable,
                    str(_project_root / "run_tracking_evaluation.py"),
                    "--images", str(img_dir),
                    "--model", "custom",
                    "--frames", str(frames),
                    "--output", str(deep_output_dir),
                ]

                print(f"\n{'='*70}")
                print(f"  深度方法 (微调): {seq_name} {'[有GT]' if has_gt else '[无GT-仅FPS]'}")
                print(f"{'='*70}")

                result = subprocess.run(cmd, capture_output=True, text=True,
                                       encoding='utf-8', errors='replace',
                                       cwd=_project_root, timeout=600)
                stats = _extract_stats_from_output(result.stdout, result.stderr)

                if result.returncode != 0:
                    print(f"[警告] 深度方法运行有错误 (returncode={result.returncode})")
                    print(result.stderr[-1500:])

                deep_output_dir = _project_root / "tracking_evaluation" / seq_name / "custom"
                if has_gt:
                    from tools.evaluate import TrackingEvaluator
                    evaluator = TrackingEvaluator(iou_threshold=0.5)
                    gt = evaluator.load_mot_gt(gt_file, class_ids=[1])

                    pred_file = deep_output_dir / "predictions.txt"
                    if pred_file.exists():
                        predictions = _load_mot_predictions(pred_file)
                        metrics_obj = evaluator.evaluate(predictions, gt)
                        metrics = metrics_obj.to_dict()
                        metrics["FPS"] = stats.get("fps", 0.0)
                        metrics["method"] = "深度方法 (微调)"
                        metrics["scene"] = seq_name
                        reporter.add_result(metrics["method"], seq_name, metrics)
                    else:
                        reporter.add_result("深度方法 (微调)", seq_name,
                            {"method": "深度方法 (微调)", "scene": seq_name,
                             "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A",
                             "IDF1": "N/A", "IDSW": "N/A", "Precision": "N/A", "Recall": "N/A"})
                else:
                    reporter.add_result("深度方法 (微调)", seq_name,
                        {"method": "深度方法 (微调)", "scene": seq_name,
                         "FPS": stats.get("fps", 0.0), "MOTA": "N/A", "MOTP": "N/A"})

        # 大模型方法 (Qwen2-VL + LoRA)
        # 注意: 大模型方法通过 visualize.py 一次性处理所有序列，
        # 因此放在 per-sequence 循环之外
        if 'large_model' in methods_to_run:
            large_model_output = Path(args.large_model_output)
            per_seq_metrics = run_large_model_method(seq_names, frames, large_model_output)
            for seq_name, metrics in per_seq_metrics.items():
                reporter.add_result(metrics["method"], seq_name, metrics)

        # 生成对比报告（MOT17模式）
        print(f"\n{'='*70}")
        print("  生成对比报告")
        print(f"{'='*70}")

        reporter.save_json_report()
        csv_path = output_dir / "comparison_table.csv"
        reporter.generate_table_csv(csv_path)
        reporter.print_summary()

        markdown_table = reporter.generate_table_markdown()
        print(f"\n{'='*70}")
        print("  Markdown 对比表格")
        print(f"{'='*70}")
        print(markdown_table)

        md_path = output_dir / "comparison_table.md"
        md_path.write_text(markdown_table, encoding='utf-8')
        print(f"\nMarkdown表格已保存: {md_path}")

        print(f"\n所有报告保存至: {output_dir}/")
        for f in sorted(output_dir.glob("*")):
            print(f"  - {f.name}")

        print(f"\n{'='*70}")
        print("  MOT17对比实验完成!")
        print(f"{'='*70}")
        return

    # 自收集视频模式（原有逻辑）
    # 计算总任务数
    total_tasks = len(args.videos_scenes) * len(methods_to_run)
    pbar = tqdm(total=total_tasks, desc="总体进度", unit="任务", ncols=100)
    
    for scene_name in args.videos_scenes:
        video_path = CUSTOM_VIDEOS_DIR / f"{scene_name}.mp4"
        if not video_path.exists():
            print(f"\n[警告] 视频不存在: {video_path}，跳过")
            continue
        
        has_gt = check_gt_available(scene_name)
        if has_gt:
            print(f"\n[信息] {scene_name}: 检测到GT标注 ({len(list(CUSTOM_LABELS_DIR.glob(f'{scene_name}_frame_*.txt')))} 帧)")
        else:
            print(f"\n[信息] {scene_name}: 无GT标注，仅收集FPS/处理时间")
        
        # 传统方法
        if 'traditional' in methods_to_run:
            pbar.set_description(f"处理 {scene_name} - 传统方法")
            metrics = run_traditional_method(video_path, scene_name, has_gt, args.frames, 
                                              use_hog_api=use_hog_api,
                                              use_svm_api=use_svm_api)
            if metrics:
                reporter.add_result(metrics["method"], scene_name, metrics)
            else:
                reporter.add_result("传统方法 (HOG+SVM)", scene_name, 
                    {"method": "传统方法 (HOG+SVM)", "scene": scene_name, 
                     "FPS": 0, "MOTA": "FAILED", "MOTP": "FAILED"})
            pbar.update(1)
        
        # 深度方法 (预训练)
        if 'deep_pretrained' in methods_to_run:
            pbar.set_description(f"处理 {scene_name} - 深度方法(原始)")
            metrics = run_deep_method(video_path, scene_name, has_gt, args.frames, 'pretrained')
            if metrics:
                reporter.add_result(metrics["method"], scene_name, metrics)
            else:
                reporter.add_result("深度方法 (原始)", scene_name,
                    {"method": "深度方法 (原始)", "scene": scene_name,
                     "FPS": 0, "MOTA": "FAILED", "MOTP": "FAILED"})
            pbar.update(1)
        
        # 深度方法 (微调)
        if 'deep_custom' in methods_to_run:
            pbar.set_description(f"处理 {scene_name} - 深度方法(微调)")
            metrics = run_deep_method(video_path, scene_name, has_gt, args.frames, 'custom')
            if metrics:
                reporter.add_result(metrics["method"], scene_name, metrics)
            else:
                reporter.add_result("深度方法 (微调)", scene_name,
                    {"method": "深度方法 (微调)", "scene": scene_name,
                     "FPS": 0, "MOTA": "FAILED", "MOTP": "FAILED"})
            pbar.update(1)
    
    pbar.close()
    
    # 生成对比报告
    print(f"\n{'='*70}")
    print("  生成对比报告")
    print(f"{'='*70}")
    
    # 保存JSON报告
    reporter.save_json_report()
    
    # 保存CSV表格
    csv_path = output_dir / "comparison_table.csv"
    reporter.generate_table_csv(csv_path)
    
    # 打印汇总
    reporter.print_summary()
    
    # Markdown表格
    markdown_table = reporter.generate_table_markdown()
    print(f"\n{'='*70}")
    print("  Markdown 对比表格")
    print(f"{'='*70}")
    print(markdown_table)
    
    md_path = output_dir / "comparison_table.md"
    md_path.write_text(markdown_table, encoding='utf-8')
    print(f"\nMarkdown表格已保存: {md_path}")
    
    # 保存独立文件
    print(f"\n所有报告保存至: {output_dir}/")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")
    
    print(f"\n{'='*70}")
    print("  对比实验完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
