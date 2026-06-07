"""
对比实验主控脚本

统一调度三种方法，收集结果，调用对比模块生成报告。

使用方法:
  # 对所有视频运行所有方法（scene5有GT标注）
  python run_comparison.py --videos-scenes scene1 scene2 scene3 scene4 scene5 --frames 150
  
  # 只运行特定方法
  python run_comparison.py --videos-scenes scene5 --methods traditional --frames 100
"""

import sys
import os
from pathlib import Path
import subprocess
import json
import argparse
import re
from typing import List, Dict, Optional

# 添加项目根目录到路径
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 固定参数
CUSTOM_VIDEOS_DIR = _project_root / "data" / "custom" / "videos"
CUSTOM_LABELS_DIR = _project_root / "data" / "yolo_custom" / "labels" / "val"
KITTI_DIR = _project_root / "data" / "kitti"
KITTI_LABELS_DIR = _project_root / "data" / "kitti" / "labels"
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
                           output_dir: Optional[Path] = None) -> Optional[Dict]:
    """
    运行传统方法 (HOG+SVM + Kalman)
    
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
    
    if has_gt:
        cmd.extend([
            "--eval",
            "--labels", str(CUSTOM_LABELS_DIR),
            "--scene", scene_name,
        ])
    
    print(f"\n{'='*70}")
    print(f"  传统方法 (HOG+SVM): {scene_name} {'[有GT]' if has_gt else '[无GT-仅FPS]'}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=_project_root, timeout=600)
    
    # 解析输出中的统计信息
    stats = _extract_stats_from_output(result.stdout, result.stderr)
    
    if result.returncode != 0:
        print(f"[警告] 传统方法运行有错误 (returncode={result.returncode})")
        print(result.stderr[-1500:])
        # 即使有错误也尝试提取已有信息
    
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
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=_project_root, timeout=600)
    
    stats = _extract_stats_from_output(result.stdout, result.stderr)
    
    if result.returncode != 0:
        print(f"[警告] 深度方法运行有错误 (returncode={result.returncode})")
        print(result.stderr[-1500:])
    
    if has_gt:
        # 读取评估文件
        expected_output_dir = _project_root / "tracking_evaluation" / video_path.stem / model_type
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='对比实验主控脚本 - 自收集多目标跟踪数据集')
    parser.add_argument('--videos-scenes', type=str, nargs='+',
                        default=['scene1', 'scene2', 'scene3', 'scene4', 'scene5'],
                        help='要测试的场景名称列表 (scene1, scene2, ...)')
    parser.add_argument('--methods', type=str, nargs='+',
                        choices=['traditional', 'deep_pretrained', 'deep_custom', 'all'],
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

    args = parser.parse_args()

    # 确定要运行的方法
    if 'all' in args.methods:
        methods_to_run = ['traditional', 'deep_pretrained', 'deep_custom']
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

    # 自收集视频模式（原有逻辑）
    for scene_name in args.videos_scenes:
        video_path = CUSTOM_VIDEOS_DIR / f"{scene_name}.mp4"
        if not video_path.exists():
            print(f"[警告] 视频不存在: {video_path}，跳过")
            continue
        
        has_gt = check_gt_available(scene_name)
        if has_gt:
            print(f"\n[信息] {scene_name}: 检测到GT标注 ({len(list(CUSTOM_LABELS_DIR.glob(f'{scene_name}_frame_*.txt')))} 帧)")
        else:
            print(f"\n[信息] {scene_name}: 无GT标注，仅收集FPS/处理时间")
        
        # 传统方法
        if 'traditional' in methods_to_run:
            metrics = run_traditional_method(video_path, scene_name, has_gt, args.frames)
            if metrics:
                reporter.add_result(metrics["method"], scene_name, metrics)
            else:
                reporter.add_result("传统方法 (HOG+SVM)", scene_name, 
                    {"method": "传统方法 (HOG+SVM)", "scene": scene_name, 
                     "FPS": 0, "MOTA": "FAILED", "MOTP": "FAILED"})
        
        # 深度方法 (预训练)
        if 'deep_pretrained' in methods_to_run:
            metrics = run_deep_method(video_path, scene_name, has_gt, args.frames, 'pretrained')
            if metrics:
                reporter.add_result(metrics["method"], scene_name, metrics)
            else:
                reporter.add_result("深度方法 (原始)", scene_name,
                    {"method": "深度方法 (原始)", "scene": scene_name,
                     "FPS": 0, "MOTA": "FAILED", "MOTP": "FAILED"})
        
        # 深度方法 (微调)
        if 'deep_custom' in methods_to_run:
            metrics = run_deep_method(video_path, scene_name, has_gt, args.frames, 'custom')
            if metrics:
                reporter.add_result(metrics["method"], scene_name, metrics)
            else:
                reporter.add_result("深度方法 (微调)", scene_name,
                    {"method": "深度方法 (微调)", "scene": scene_name,
                     "FPS": 0, "MOTA": "FAILED", "MOTP": "FAILED"})
    
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
