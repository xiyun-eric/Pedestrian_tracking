"""
Phase 5: MOT 标准评估

实现 MOTA/MOTP/IDF1/IDSW/MT/ML/FP/FN 等 MOT 标准指标计算
使用 motmetrics 库进行评估

流程:
  1. 加载 Qwen2-VL 模型 (可选 LoRA 权重)
  2. 对 MOT17 序列滑动窗口推理
  3. 跨窗口 ID 关联 (IoU 匹配)
  4. 保存为 MOT 标准格式
  5. 使用 motmetrics 计算指标

使用方式:
  # 推理并评估指定 LoRA 权重
  python model_method/evaluate.py --lora-path runs/stage1/final

  # 评估原始模型 (无 LoRA)
  python model_method/evaluate.py --no-lora

  # 快速模式 (只评估前 20 帧)
  python model_method/evaluate.py --lora-path runs/stage1/final --quick

  # 只评估已有结果文件 (不重新推理)
  python model_method/evaluate.py --eval-only --result-dir runs/mot_results

  # 指定序列和窗口参数
  python model_method/evaluate.py --lora-path runs/stage1/final \
      --sequences MOT17-02-FRCNN --window-size 2 --stride 1
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "Qwen"
MOT17_PATH = PROJECT_ROOT / "data" / "MOT17"
OUTPUT_DIR = PROJECT_ROOT / "runs"

# 修复 Windows GBK 编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# 复用现有模块的函数
# ============================================================

from model_method.visualize import (
    parse_bboxes_from_text,
    parse_detect_match_response,
    qwen_bbox_to_pixel,
    compute_iou,
)
from model_method.build_sft_data import get_sequence_info


# ============================================================
# 跟踪结果解析 (支持多帧格式)
# ============================================================

def parse_tracks_from_response(response, num_frames):
    """
    从模型输出中解析跟踪结果, 支持多帧格式

    格式:
      Frame 1:
      ID1: <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
      ID2: ...

      Frame 2:
      ID1: ...
      ID2: ...

    Args:
        response: 模型输出文本
        num_frames: 窗口中的帧数

    Returns:
        tracks: {local_id: {frame_idx(0-based): (x1, y1, x2, y2)}}
                bbox 为 Qwen 0-1000 坐标
    """
    tracks = defaultdict(dict)

    # 尝试按 "Frame N:" 分割
    frame_pattern = r'Frame\s*(\d+)\s*:'
    frame_sections = re.split(frame_pattern, response)

    if len(frame_sections) > 1:
        # 有 Frame 标记: [pre, '1', content1, '2', content2, ...]
        for i in range(1, len(frame_sections), 2):
            if i + 1 >= len(frame_sections):
                break
            try:
                frame_num = int(frame_sections[i].strip())
            except ValueError:
                continue
            frame_idx = frame_num - 1  # 转为 0-based
            if frame_idx < 0 or frame_idx >= num_frames:
                continue

            content = frame_sections[i + 1]
            _parse_id_bboxes_from_section(content, tracks, frame_idx)
    else:
        # 无 Frame 标记: 尝试按行解析, 每行一个 ID
        # 假设模型按帧顺序输出, 每个 ID 出现 num_frames 次
        lines = response.strip().split('\n')
        id_line_count = defaultdict(int)
        for line in lines:
            id_match = re.search(r'ID(\d+)\s*:', line)
            if not id_match:
                continue
            track_id = int(id_match.group(1))
            bboxes = parse_bboxes_from_text(line)
            if bboxes:
                frame_idx = id_line_count[track_id]
                if frame_idx < num_frames:
                    tracks[track_id][frame_idx] = bboxes[0]
                    id_line_count[track_id] += 1

    return dict(tracks)


def _parse_id_bboxes_from_section(section, tracks, frame_idx):
    """从一段文本中解析 ID 和 bbox, 添加到 tracks"""
    lines = section.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        id_match = re.search(r'ID(\d+)\s*:', line)
        if not id_match:
            continue
        track_id = int(id_match.group(1))
        bboxes = parse_bboxes_from_text(line)
        if bboxes:
            # 每个帧每个 ID 只取第一个 bbox
            if frame_idx not in tracks[track_id]:
                tracks[track_id][frame_idx] = bboxes[0]


def parse_detect_match_tracks(response, num_frames=2):
    """
    解析"先检测再匹配"格式的输出, 转换为tracks格式

    Returns:
        tracks: {local_id: {frame_idx(0-based): (x1, y1, x2, y2)}}
                bbox 为 Qwen 0-1000 坐标
    """
    f1_bboxes, f2_bboxes, matching = parse_detect_match_response(response)

    tracks = defaultdict(dict)

    # 为每对匹配分配一个local_id
    for local_id, (i, j) in enumerate(matching):
        if i < len(f1_bboxes):
            tracks[local_id][0] = f1_bboxes[i]  # Frame 1
        if j < len(f2_bboxes):
            tracks[local_id][1] = f2_bboxes[j]  # Frame 2

    # 未匹配的检测框也加入tracks (只在一帧出现)
    matched_f1 = {i for i, j in matching}
    matched_f2 = {j for i, j in matching}

    next_id = len(matching)
    for i, bbox in enumerate(f1_bboxes):
        if i not in matched_f1:
            tracks[next_id][0] = bbox
            next_id += 1

    for j, bbox in enumerate(f2_bboxes):
        if j not in matched_f2:
            tracks[next_id][1] = bbox
            next_id += 1

    return dict(tracks)


# ============================================================
# MOT 结果文件 I/O
# ============================================================

def save_mot_results(tracks_per_frame, output_path):
    """
    保存为 MOT 标准格式: frame,id,x,y,w,h,conf,-1,-1,-1

    Args:
        tracks_per_frame: Dict[int, List[Tuple[int, bbox]]]
            {frame_id: [(track_id, [x1, y1, x2, y2]), ...]}
            bbox 为像素坐标
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = sum(len(v) for v in tracks_per_frame.values())
    with open(output_path, 'w') as f:
        for frame_id in sorted(tracks_per_frame.keys()):
            for track_id, bbox in tracks_per_frame[frame_id]:
                x1, y1, x2, y2 = bbox
                x, y, w, h = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
                f.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

    print(f"  保存 MOT 结果: {output_path} ({total} 条)")


# ============================================================
# MOT 评估器
# ============================================================

class MOTEvaluator:
    """MOT17 标准评估器 (基于 motmetrics)"""

    # 评估使用的指标
    METRICS = [
        'mota', 'motp', 'idf1',
        'idp', 'idr',
        'num_switches',          # IDSW
        'num_false_positives',   # FP
        'num_misses',            # FN
        'num_matches',           # TP
        'num_detections',        # 总预测数
        'num_objects',           # 总 GT 数
        'mostly_tracked',        # MT
        'mostly_lost',           # ML
        'partially_tracked',     # PT
        'num_fragmentations',    # Frag
        'precision', 'recall',
    ]

    # 指标名称映射 (用于显示)
    NAME_MAP = {
        'mota': 'MOTA', 'motp': 'MOTP', 'idf1': 'IDF1',
        'idp': 'IDP', 'idr': 'IDR',
        'num_switches': 'IDSW',
        'num_false_positives': 'FP',
        'num_misses': 'FN',
        'num_matches': 'TP',
        'num_detections': 'Pred',
        'num_objects': 'GT',
        'mostly_tracked': 'MT',
        'mostly_lost': 'ML',
        'partially_tracked': 'PT',
        'num_fragmentations': 'Frag',
        'precision': 'Prec',
        'recall': 'Reca',
    }

    def __init__(self):
        try:
            import motmetrics as mm
            self.mm = mm
        except ImportError:
            print("❌ motmetrics 未安装, 请运行: pip install motmetrics")
            sys.exit(1)

        self.mh = mm.metrics.create()

    def _load_mot17_gt(self, gt_path):
        """
        加载 MOT17 GT, 过滤非行人

        MOT17 GT 格式: frame,id,x,y,w,h,conf,class,vis
        只保留 class==1 (pedestrian) 且 conf>=1
        """
        mm = self.mm

        rows = []
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue
                frame = int(parts[0])
                tid = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                conf = float(parts[6])
                cls = int(parts[7])

                if cls == 1 and conf >= 1:
                    rows.append((frame, tid, x, y, w, h))

        if not rows:
            return mm.io.loadtxt('', fmt='mot15-2D')

        # 写入临时文件让 motmetrics 加载
        import tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.txt', delete=False, encoding='utf-8'
            ) as f:
                for frame, tid, x, y, w, h in rows:
                    f.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
                tmp_path = f.name

            gt = mm.io.loadtxt(tmp_path, fmt='mot15-2D', min_confidence=1)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return gt

    def evaluate_sequence(self, gt_path, result_path):
        """
        评估单个序列

        Returns:
            acc: MOTAccumulator, 或 None (失败时)
        """
        mm = self.mm

        gt = self._load_mot17_gt(gt_path)
        ts = mm.io.loadtxt(str(result_path), fmt='mot15-2D', min_confidence=0)

        if len(gt) == 0:
            print(f"    ⚠️ GT 为空: {gt_path}")
            return None

        if len(ts) == 0:
            print(f"    ⚠️ 结果为空: {result_path}")
            return None

        # IoU 阈值 0.5 (MOT 标准)
        acc = mm.utils.compare_to_groundtruth(gt, ts, dist='iou', distth=0.5)
        return acc

    def evaluate_mot17(self, result_dir, gt_dir, sequences=None):
        """
        评估 MOT17 多个序列

        Returns:
            summary: 汇总 DataFrame (含 overall 行)
            names: 评估的序列名列表
        """
        if sequences is None:
            sequences = sorted([
                d for d in os.listdir(gt_dir)
                if (Path(gt_dir) / d).is_dir()
                and (Path(gt_dir) / d / 'gt' / 'gt.txt').exists()
            ])

        accumulators = []
        names = []

        for seq_name in sequences:
            gt_path = Path(gt_dir) / seq_name / 'gt' / 'gt.txt'
            result_path = Path(result_dir) / f"{seq_name}.txt"

            if not gt_path.exists():
                print(f"  ⚠️ GT 不存在: {gt_path}")
                continue
            if not result_path.exists():
                print(f"  ⚠️ 结果不存在: {result_path}")
                continue

            print(f"  评估 {seq_name}...")
            acc = self.evaluate_sequence(gt_path, result_path)
            if acc is not None:
                accumulators.append(acc)
                names.append(seq_name)

        if not accumulators:
            print("  ❌ 没有可评估的序列")
            return None, []

        summary = self.mh.compute_many(
            accumulators,
            metrics=self.METRICS,
            names=names,
            generate_overall=True,
        )

        return summary, names

    def render_summary(self, summary):
        """渲染汇总表为字符串"""
        return self.mm.io.render_summary(
            summary,
            formatters=self.mh.formatters,
            namemap=self.NAME_MAP,
        )

    def summary_to_dict(self, summary):
        """将汇总表转为字典 (便于 JSON 序列化)"""
        result = {}
        try:
            df = summary
            for idx in df.index:
                idx_name = str(idx)
                result[idx_name] = {}
                for col in df.columns:
                    val = df.loc[idx, col]
                    # 转换 numpy 类型
                    if hasattr(val, 'item'):
                        val = val.item()
                    if isinstance(val, float):
                        if np.isnan(val):
                            val = None
                        else:
                            val = round(val, 4)
                    result[idx_name][col] = val
        except Exception as e:
            print(f"  ⚠️ summary_to_dict 转换失败: {e}")
        return result


# ============================================================
# Qwen2-VL 跟踪推理器
# ============================================================

class Qwen2VLTracker:
    """Qwen2-VL 行人跟踪推理器"""

    def __init__(self, model_path, lora_path=None, max_pixels=None):
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        print(f"  加载模型: {model_path}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=torch.float16,
            device_map="auto",
        )

        if lora_path:
            from peft import PeftModel
            print(f"  加载 LoRA: {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.processor = AutoProcessor.from_pretrained(str(model_path))

        if max_pixels is not None:
            self.processor.image_processor.max_pixels = max_pixels
            self.processor.image_processor.min_pixels = max_pixels // 4

        self.torch = torch
        self.model.eval()

    def track_window(self, image_paths, prompt=None, fmt='track'):
        """
        对一个窗口的帧进行跟踪

        Args:
            image_paths: 图像路径列表
            prompt: 自定义提示词 (None=使用默认)
            fmt: 格式类型 ('track'=传统跟踪格式, 'detect_match'=先检测再匹配)

        Returns:
            response: 模型原始输出文本
        """
        from qwen_vl_utils import process_vision_info

        num_frames = len(image_paths)

        if prompt is None:
            if fmt == 'detect_match':
                prompt = (
                    "Detect all pedestrians in these 2 consecutive frames, then match them across frames.\n\n"
                    "Step 1 - Detection: For each frame, output all pedestrian bounding boxes (one per line).\n"
                    "Format: <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>\n\n"
                    "Step 2 - Matching: Output the correspondence between Frame 1 and Frame 2 detections.\n"
                    "Format: Matching: i-j (where i is the index in Frame 1, j is the index in Frame 2)\n\n"
                    "Coordinates are normalized to 0-1000."
                )
            else:
                prompt = (
                    f"Track all pedestrians across these {num_frames} frames. "
                    "For each person, assign a unique tracking ID and provide their "
                    "bounding box in each frame. "
                    "Output format per frame: Frame N: IDx: "
                    "<|object_ref_start|>person<|object_ref_end|>"
                    "<|box_start|>(x1,y1),(x2,y2)<|box_end|> "
                    "Coordinates are normalized to 0-1000."
                )

        content = []
        for path in image_paths:
            content.append({"type": "image", "image": path})
        content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.tokenizer.decode(
            generated_ids[0], skip_special_tokens=False
        )

        return response


# ============================================================
# 滑动窗口推理 + 跨窗口 ID 关联
# ============================================================

def run_inference_on_sequence(
    tracker, seq_path, img_width, img_height,
    window_size=2, stride=1, max_frames=None,
    iou_threshold=0.3, max_inactive=30, fmt='track',
):
    """
    对一个 MOT17 序列运行推理

    使用滑动窗口 + IoU 匹配实现跨窗口 ID 关联:
      1. 滑动窗口 [i, i+1, ...] 推理
      2. 用第一帧与上一窗口的最后一帧做 IoU 匹配
      3. 匹配的复用 global ID, 未匹配的分配新 ID

    Args:
        tracker: Qwen2VLTracker 实例
        seq_path: 序列目录
        img_width, img_height: 图像尺寸
        window_size: 窗口大小 (帧数)
        stride: 滑动步长
        max_frames: 最大帧数 (None=全部)
        iou_threshold: ID 匹配的 IoU 阈值
        max_inactive: 轨迹最大不活跃帧数 (超过则终止)

    Returns:
        tracks_per_frame: Dict[int, List[Tuple[int, bbox]]]
            {frame_id: [(track_id, [x1, y1, x2, y2]), ...]}
            bbox 为像素坐标
    """
    img_dir = seq_path / "img1"
    frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    if max_frames is not None:
        frames = frames[:max_frames]

    total_frames = len(frames)
    num_windows = max(0, (total_frames - window_size) // stride + 1)

    print(f"    总帧数: {total_frames}, 窗口: {window_size}, 步长: {stride}, "
          f"共 {num_windows} 个窗口")

    # 全局轨迹: {global_id: {frame_id: bbox_pixel}}
    global_tracks = defaultdict(dict)
    next_global_id = 1

    # 活跃轨迹: {global_id: (last_frame, last_bbox)}
    active_tracks = {}

    total_inference_time = 0.0
    total_detections = 0

    for win_idx, start_idx in enumerate(range(0, total_frames - window_size + 1, stride)):
        window_frames = frames[start_idx:start_idx + window_size]
        image_paths = [str(img_dir / f) for f in window_frames]

        # 帧号 (1-indexed, 与 MOT 格式一致)
        frame_ids = [start_idx + i + 1 for i in range(window_size)]

        # 推理
        t0 = time.time()
        try:
            response = tracker.track_window(image_paths, fmt=fmt)
        except Exception as e:
            print(f"    ⚠️ 窗口 {win_idx + 1} 推理失败: {e}")
            continue
        t1 = time.time()
        total_inference_time += (t1 - t0)

        # 解析跟踪结果
        if fmt == 'detect_match':
            local_tracks = parse_detect_match_tracks(response, window_size)
        else:
            local_tracks = parse_tracks_from_response(response, window_size)

        # 转换为像素坐标
        # local_tracks_pixel: {local_id: {frame_id: [px1, py1, px2, py2]}}
        local_tracks_pixel = {}
        for local_id, frame_bboxes in local_tracks.items():
            converted = {}
            for frame_idx, bbox in frame_bboxes.items():
                x1, y1, x2, y2 = bbox
                px1, py1, px2, py2 = qwen_bbox_to_pixel(
                    x1, y1, x2, y2, img_width, img_height
                )
                fid = frame_ids[frame_idx]
                converted[fid] = [px1, py1, px2, py2]
            if converted:
                local_tracks_pixel[local_id] = converted

        if not local_tracks_pixel:
            if (win_idx + 1) % 10 == 0 or win_idx == 0:
                print(f"    窗口 {win_idx + 1}/{num_windows}: 无有效检测")
            continue

        # ID 匹配: 用第一帧与活跃轨迹匹配
        local_to_global = {}
        first_frame_id = frame_ids[0]

        if active_tracks:
            # 收集第一帧的 local 检测
            first_frame_detections = []
            for local_id, frame_bboxes in local_tracks_pixel.items():
                if first_frame_id in frame_bboxes:
                    first_frame_detections.append(
                        (local_id, frame_bboxes[first_frame_id])
                    )

            # 计算所有 (local, global) 对的 IoU
            matches = []
            for local_id, local_bbox in first_frame_detections:
                for global_id, (last_frame, last_bbox) in active_tracks.items():
                    iou = compute_iou(local_bbox, last_bbox)
                    if iou > iou_threshold:
                        matches.append((iou, local_id, global_id))

            # 按 IoU 降序贪心匹配
            matches.sort(key=lambda x: -x[0])
            used_locals = set()
            used_globals = set()
            for iou, local_id, global_id in matches:
                if local_id not in used_locals and global_id not in used_globals:
                    local_to_global[local_id] = global_id
                    used_locals.add(local_id)
                    used_globals.add(global_id)

        # 为未匹配的 local ID 分配新 global ID
        for local_id in local_tracks_pixel:
            if local_id not in local_to_global:
                local_to_global[local_id] = next_global_id
                next_global_id += 1

        # 更新全局轨迹和活跃轨迹
        for local_id, frame_bboxes in local_tracks_pixel.items():
            global_id = local_to_global[local_id]
            for fid, bbox in frame_bboxes.items():
                global_tracks[global_id][fid] = bbox
                active_tracks[global_id] = (fid, bbox)
                total_detections += 1

        # 清理过期活跃轨迹
        last_frame_id = frame_ids[-1]
        active_tracks = {
            gid: (f, b) for gid, (f, b) in active_tracks.items()
            if last_frame_id - f <= max_inactive
        }

        # 进度日志
        if (win_idx + 1) % 10 == 0 or win_idx == 0:
            avg_time = total_inference_time / (win_idx + 1)
            print(f"    窗口 {win_idx + 1}/{num_windows}: "
                  f"avg {avg_time:.2f}s/窗口, "
                  f"活跃轨迹 {len(active_tracks)}, "
                  f"累计检测 {total_detections}")

    # 转换为 MOT 格式: {frame_id: [(track_id, bbox), ...]}
    tracks_per_frame = defaultdict(list)
    for global_id, frame_bboxes in global_tracks.items():
        for fid, bbox in frame_bboxes.items():
            tracks_per_frame[fid].append((global_id, bbox))

    avg_time = total_inference_time / max(1, num_windows)
    print(f"    推理完成: {len(global_tracks)} 个轨迹, "
          f"{total_detections} 个检测, 平均 {avg_time:.2f}s/窗口")

    return dict(tracks_per_frame)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: MOT 标准评估")
    parser.add_argument(
        '--lora-path', type=str, default=None,
        help='LoRA 权重路径 (不指定且未 --no-lora 则使用 runs/stage1/final)'
    )
    parser.add_argument(
        '--no-lora', action='store_true',
        help='评估原始模型 (无 LoRA)'
    )
    parser.add_argument(
        '--model-path', type=str, default=str(MODEL_PATH),
        help='基座模型路径'
    )
    parser.add_argument(
        '--mot17-root', type=str, default=str(MOT17_PATH),
        help='MOT17 数据集根目录'
    )
    parser.add_argument(
        '--result-dir', type=str, default=None,
        help='MOT 结果输出目录 (默认: output/mot_results/<lora_name>)'
    )
    parser.add_argument(
        '--report-path', type=str, default=None,
        help='评估报告输出路径 (默认: output/mot_results/<lora_name>/report.txt)'
    )
    parser.add_argument(
        '--sequences', type=str, default=None,
        help='要评估的序列 (逗号分隔, 如 MOT17-02-FRCNN,MOT17-04-FRCNN)'
    )
    parser.add_argument(
        '--window-size', type=int, default=2,
        help='滑动窗口大小 (帧数)'
    )
    parser.add_argument(
        '--stride', type=int, default=1,
        help='滑动步长'
    )
    parser.add_argument(
        '--max-frames', type=int, default=None,
        help='每个序列最大帧数 (None=全部)'
    )
    parser.add_argument(
        '--max-pixels', type=int, default=28 * 28 * 2,
        help='图像最大像素数 (降低分辨率以节省显存)'
    )
    parser.add_argument(
        '--iou-threshold', type=float, default=0.3,
        help='跨窗口 ID 匹配的 IoU 阈值'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='快速模式: 只评估前 20 帧'
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='只评估已有结果文件, 不重新推理'
    )
    parser.add_argument(
        '--format', type=str, default='track', choices=['track', 'detect_match'],
        help='推理格式: track=传统跟踪格式, detect_match=先检测再匹配 (默认: track)'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5: MOT 标准评估")
    print("=" * 60)

    if args.quick:
        args.max_frames = 20
        print("  ⚡ Quick 模式: 只评估前 20 帧")

    # 确定 LoRA 路径和结果目录
    if args.no_lora:
        lora_path = None
        lora_name = "no_lora"
    elif args.lora_path:
        lora_path = args.lora_path
        lora_name = Path(args.lora_path).name
    else:
        # 默认使用 stage1 权重
        default_lora = OUTPUT_DIR / "stage1" / "final"
        if default_lora.exists():
            lora_path = str(default_lora)
            lora_name = "stage1_final"
        else:
            lora_path = None
            lora_name = "no_lora"

    if args.result_dir is None:
        args.result_dir = str(OUTPUT_DIR / "mot_results" / lora_name)
    if args.report_path is None:
        args.report_path = str(Path(args.result_dir) / "report.txt")

    print(f"  LoRA: {lora_path or 'None (原始模型)'}")
    print(f"  结果目录: {args.result_dir}")
    print(f"  报告路径: {args.report_path}")
    print()

    # 序列列表
    if args.sequences:
        sequences = [s.strip() for s in args.sequences.split(',')]
    else:
        sequences = sorted([
            d for d in os.listdir(args.mot17_root)
            if (Path(args.mot17_root) / d).is_dir()
            and (Path(args.mot17_root) / d / 'gt' / 'gt.txt').exists()
        ])

    print(f"  序列: {sequences}")
    print(f"  窗口: {args.window_size}, 步长: {args.stride}")
    print()

    # ============================================================
    # 步骤 1: 推理 (如果需要)
    # ============================================================

    if not args.eval_only:
        # 加载模型
        tracker = Qwen2VLTracker(
            model_path=args.model_path,
            lora_path=lora_path,
            max_pixels=args.max_pixels,
        )

        # 对每个序列推理
        for seq_name in sequences:
            print(f"\n{'=' * 60}")
            print(f"推理序列: {seq_name}")
            print(f"{'=' * 60}")

            seq_path = Path(args.mot17_root) / seq_name
            if not seq_path.exists():
                print(f"  ⚠️ 序列不存在: {seq_path}")
                continue

            info = get_sequence_info(seq_path)
            img_width = int(info.get('imWidth', 1920))
            img_height = int(info.get('imHeight', 1080))
            print(f"  图像尺寸: {img_width}x{img_height}")

            # 推理
            tracks = run_inference_on_sequence(
                tracker=tracker,
                seq_path=seq_path,
                img_width=img_width,
                img_height=img_height,
                window_size=args.window_size,
                stride=args.stride,
                max_frames=args.max_frames,
                iou_threshold=args.iou_threshold,
                fmt=args.format,
            )

            # 保存
            result_path = Path(args.result_dir) / f"{seq_name}.txt"
            save_mot_results(tracks, result_path)

        # 清理显存
        del tracker
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # 步骤 2: 评估
    # ============================================================
    print(f"\n{'=' * 60}")
    print("MOT 标准评估")
    print(f"{'=' * 60}")

    evaluator = MOTEvaluator()
    summary, names = evaluator.evaluate_mot17(
        result_dir=args.result_dir,
        gt_dir=args.mot17_root,
        sequences=sequences,
    )

    if summary is None:
        print("  ❌ 评估失败")
        return

    # 渲染结果
    report = evaluator.render_summary(summary)
    print("\n" + report)

    # 保存报告
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_path, 'w', encoding='utf-8') as f:
        f.write("MOT 评估报告\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"LoRA: {lora_path or 'None (原始模型)'}\n")
        f.write(f"序列: {names}\n")
        f.write(f"窗口大小: {args.window_size}, 步长: {args.stride}\n")
        f.write(f"最大帧数: {args.max_frames or '全部'}\n")
        f.write(f"IoU 匹配阈值: {args.iou_threshold}\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(report)

    print(f"\n  报告已保存: {args.report_path}")

    # 保存为 JSON (便于消融实验比较)
    json_path = str(Path(args.report_path).with_suffix('.json'))
    summary_dict = evaluator.summary_to_dict(summary)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'lora_path': lora_path,
            'lora_name': lora_name,
            'sequences': names,
            'window_size': args.window_size,
            'stride': args.stride,
            'max_frames': args.max_frames,
            'iou_threshold': args.iou_threshold,
            'metrics': summary_dict,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"  JSON 结果已保存: {json_path}")


if __name__ == '__main__':
    main()
