"""
Phase 2: MOT17 → Qwen2-VL SFT 数据构建脚本

方案 B (温和稳定): 回归原生检测格式 + 后处理跟踪
  - Stage 1: 单帧检测数据 (原生 Qwen2-VL 格式, 利用模型已有能力)
  - Stage 2: 双帧跟踪数据 (用于后续 ID 跟踪格式学习或 RL 精调)

输出格式: JSONL, 每行一个样本
  {"messages": [{"role": "user", "content": [...]}, {"role": "assistant", "content": [...]}]}
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

# 项目根目录 (Pedestrian_tracking/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOT17_ROOT = PROJECT_ROOT / "data" / "MOT17"
OUTPUT_DIR = PROJECT_ROOT / "data"

# 修复 Windows GBK 编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# 坐标转换: MOT17 像素坐标 → Qwen2-VL 0-1000 坐标
# ============================================================

def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
    """Qwen2-VL 官方 resize 函数, 计算模型内部 resize 后的图像尺寸"""
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def mot17_to_qwen_bbox(x, y, w, h, orig_height, orig_width):
    """
    将 MOT17 像素坐标 [x, y, w, h] 转换为 Qwen2-VL 0-1000 坐标 (x1,y1),(x2,y2)

    Qwen2-VL 的 bbox 坐标是 resize 后图像上的绝对坐标, 归一化到 0-1000
    """
    new_h, new_w = smart_resize(orig_height, orig_width)
    scale_w = new_w / orig_width
    scale_h = new_h / orig_height

    x1 = round(x * scale_w)
    y1 = round(y * scale_h)
    x2 = round((x + w) * scale_w)
    y2 = round((y + h) * scale_h)

    # 归一化到 0-1000
    x1 = int(x1 / new_w * 1000)
    y1 = int(y1 / new_h * 1000)
    x2 = int(x2 / new_w * 1000)
    y2 = int(y2 / new_h * 1000)

    # clamp 到有效范围
    x1 = max(0, min(x1, 999))
    y1 = max(0, min(y1, 999))
    x2 = max(x1 + 1, min(x2, 1000))
    y2 = max(y1 + 1, min(y2, 1000))

    return (x1, y1), (x2, y2)


# ============================================================
# MOT17 GT 解析
# ============================================================

def parse_gt(gt_path):
    """
    解析 MOT17 GT 文件

    格式: frame,id,x,y,w,h,conf,class,visibility
    - conf: 0 表示 ignore, 1 表示 active
    - class: 1=pedestrian, 2=static person, 7=ignored, ...
    - 只保留 class==1 且 conf>=1 的行人
    """
    annotations = defaultdict(list)
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6])
            class_id = int(parts[7])
            visibility = float(parts[8])

            # 只保留行人 (class==1) 且有效 (conf>=1)
            if class_id == 1 and conf >= 1:
                annotations[frame_id].append({
                    'id': track_id,
                    'bbox': [x, y, w, h],
                    'visibility': visibility,
                })
    return annotations


def get_sequence_info(seq_path):
    """读取序列信息 (图像宽高, 帧数等)"""
    seqinfo_path = seq_path / "seqinfo.ini"
    info = {}
    if seqinfo_path.exists():
        with open(seqinfo_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    info[key.strip()] = val.strip()
    return info


# ============================================================
# Stage 1: 单帧检测数据
# ============================================================

def build_stage1_samples(mot17_root, min_objects=1, max_objects=30, stride=10):
    """
    构建 Stage 1 单帧检测样本

    每个样本:
      输入: 单帧图像 + 检测提示词
      输出: 所有行人的 bbox (无 track ID, 仅检测)
    """
    samples = []
    seq_dir = mot17_root

    for seq_name in sorted(os.listdir(seq_dir)):
        seq_path = seq_dir / seq_name
        if not seq_path.is_dir():
            continue

        img_dir = seq_path / "img1"
        gt_path = seq_path / "gt" / "gt.txt"

        if not gt_path.exists() or not img_dir.exists():
            continue

        print(f"  处理序列: {seq_name}")

        # 读取序列信息
        info = get_sequence_info(seq_path)
        img_width = int(info.get('imWidth', 1920))
        img_height = int(info.get('imHeight', 1080))

        # 解析 GT
        annotations = parse_gt(gt_path)

        # 获取帧列表
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        total_frames = len(frames)

        # 按 stride 采样
        for fi in range(0, total_frames, stride):
            frame_id = fi + 1
            frame_anns = annotations.get(frame_id, [])

            if len(frame_anns) < min_objects or len(frame_anns) > max_objects:
                continue

            # 构建用户消息
            image_path = str(img_dir / frames[fi])
            prompt = (
                "Detect all pedestrians in this image. "
                "For each person, output their bounding box. "
                "Use format: <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>"
            )

            user_content = [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ]

            # 构建助手回复 (GT)
            response_parts = []
            for ann in frame_anns:
                x, y, w, h = ann['bbox']
                (x1, y1), (x2, y2) = mot17_to_qwen_bbox(x, y, w, h, img_height, img_width)
                response_parts.append(
                    f"<|object_ref_start|>person<|object_ref_end|><|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"
                )

            assistant_content = [
                {"type": "text", "text": "\n".join(response_parts)}
            ]

            sample = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "metadata": {
                    "source": seq_name,
                    "frame_id": frame_id,
                    "num_objects": len(frame_anns),
                    "stage": 1,
                }
            }
            samples.append(sample)

    return samples


# ============================================================
# Stage 2: 双帧跟踪数据
# ============================================================

def build_stage2_samples(mot17_root, stride=10, min_tracks=1):
    """
    构建 Stage 2 双帧跟踪样本

    每个样本:
      输入: 连续 2 帧图像 + 跟踪提示词
      输出: 每帧中每个行人的 track_id + bbox
    """
    samples = []
    seq_len = 2
    frame_gap = 1  # 相邻帧间隔

    seq_dir = mot17_root

    for seq_name in sorted(os.listdir(seq_dir)):
        seq_path = seq_dir / seq_name
        if not seq_path.is_dir():
            continue

        img_dir = seq_path / "img1"
        gt_path = seq_path / "gt" / "gt.txt"

        if not gt_path.exists() or not img_dir.exists():
            continue

        print(f"  处理序列: {seq_name}")

        info = get_sequence_info(seq_path)
        img_width = int(info.get('imWidth', 1920))
        img_height = int(info.get('imHeight', 1080))

        annotations = parse_gt(gt_path)
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        total_frames = len(frames)

        # 滑动窗口
        for start_idx in range(0, total_frames - seq_len * frame_gap, stride):
            frame_indices = [start_idx + i * frame_gap for i in range(seq_len)]

            # 收集每帧标注
            sample_frames = []
            for fi in frame_indices:
                if fi >= total_frames:
                    break
                frame_id = fi + 1
                frame_anns = annotations.get(frame_id, [])
                sample_frames.append({
                    'frame_id': frame_id,
                    'image_path': str(img_dir / frames[fi]),
                    'annotations': frame_anns,
                })

            if len(sample_frames) < seq_len:
                continue

            # 仅保留在序列中至少出现 2 帧的目标
            track_counts = defaultdict(int)
            for f in sample_frames:
                for ann in f['annotations']:
                    track_counts[ann['id']] += 1
            valid_tracks = {tid for tid, cnt in track_counts.items() if cnt >= 2}

            if len(valid_tracks) < min_tracks:
                continue

            # 构建用户消息
            user_content = []
            for frame in sample_frames:
                user_content.append({"type": "image", "image": frame['image_path']})

            prompt = (
                "Track all pedestrians across these 2 consecutive frames. "
                "For each person, assign a unique tracking ID and provide their "
                "bounding box in each frame. "
                "Output format per frame: Frame N: IDx: <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|> "
                "Coordinates are normalized to 0-1000."
            )
            user_content.append({"type": "text", "text": prompt})

            # 构建助手回复
            response_parts = []
            for i, frame in enumerate(sample_frames):
                frame_parts = [f"Frame {i + 1}:"]
                for ann in frame['annotations']:
                    if ann['id'] not in valid_tracks:
                        continue
                    x, y, w, h = ann['bbox']
                    (x1, y1), (x2, y2) = mot17_to_qwen_bbox(x, y, w, h, img_height, img_width)
                    frame_parts.append(
                        f"ID{ann['id']}: <|object_ref_start|>person<|object_ref_end|><|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"
                    )
                response_parts.append("\n".join(frame_parts))

            assistant_content = [
                {"type": "text", "text": "\n\n".join(response_parts)}
            ]

            sample = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "metadata": {
                    "source": seq_name,
                    "frame_ids": [f['frame_id'] for f in sample_frames],
                    "num_tracks": len(valid_tracks),
                    "stage": 2,
                }
            }
            samples.append(sample)

    return samples


# ============================================================
# Stage 2: 先检测再匹配 (Detect-then-Match) 数据
# ============================================================

def build_detect_match_samples(mot17_root, stride=10, min_tracks=1):
    """
    构建"先检测再匹配"格式的双帧样本

    格式:
      Frame 1:
      <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
      <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>

      Frame 2:
      <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
      <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>

      Matching: 0-0, 1-1, 2-2

    关键: 利用MOT17 GT中的track_id构建匹配关系
    - 同一track_id在两帧的bbox = 匹配对
    - Frame1和Frame2按相同track_id顺序输出bbox
    - 因此Matching就是 0-0, 1-1, 2-2, ...
    """
    samples = []
    seq_len = 2
    frame_gap = 1  # 相邻帧间隔

    seq_dir = mot17_root

    for seq_name in sorted(os.listdir(seq_dir)):
        seq_path = seq_dir / seq_name
        if not seq_path.is_dir():
            continue

        img_dir = seq_path / "img1"
        gt_path = seq_path / "gt" / "gt.txt"

        if not gt_path.exists() or not img_dir.exists():
            continue

        print(f"  处理序列: {seq_name}")

        info = get_sequence_info(seq_path)
        img_width = int(info.get('imWidth', 1920))
        img_height = int(info.get('imHeight', 1080))

        annotations = parse_gt(gt_path)
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        total_frames = len(frames)

        # 滑动窗口
        for start_idx in range(0, total_frames - seq_len * frame_gap, stride):
            frame_indices = [start_idx + i * frame_gap for i in range(seq_len)]

            # 收集每帧标注
            sample_frames = []
            for fi in frame_indices:
                if fi >= total_frames:
                    break
                frame_id = fi + 1
                frame_anns = annotations.get(frame_id, [])
                sample_frames.append({
                    'frame_id': frame_id,
                    'image_path': str(img_dir / frames[fi]),
                    'annotations': frame_anns,
                })

            if len(sample_frames) < seq_len:
                continue

            # 找出两帧都出现的track_id (这些是可匹配的目标)
            frame1_ids = {ann['id'] for ann in sample_frames[0]['annotations']}
            frame2_ids = {ann['id'] for ann in sample_frames[1]['annotations']}
            matched_ids = frame1_ids & frame2_ids  # 交集

            if len(matched_ids) < min_tracks:
                continue

            # 按track_id排序, 保证Frame1和Frame2的顺序一致
            sorted_ids = sorted(matched_ids)

            # 构建用户消息
            user_content = []
            for frame in sample_frames:
                user_content.append({"type": "image", "image": frame['image_path']})

            prompt = (
                "Detect all pedestrians in these 2 consecutive frames, then match them across frames.\n\n"
                "Step 1 - Detection: For each frame, output all pedestrian bounding boxes (one per line).\n"
                "Format: <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>\n\n"
                "Step 2 - Matching: Output the correspondence between Frame 1 and Frame 2 detections.\n"
                "Format: Matching: i-j (where i is the index in Frame 1, j is the index in Frame 2)\n\n"
                "Coordinates are normalized to 0-1000."
            )
            user_content.append({"type": "text", "text": prompt})

            # 构建助手回复: 先检测再匹配
            response_parts = []

            # Frame 1 检测 (只输出匹配的目标, 按sorted_ids顺序)
            frame1_lines = ["Frame 1:"]
            for tid in sorted_ids:
                # 找到该track_id在frame1的标注
                for ann in sample_frames[0]['annotations']:
                    if ann['id'] == tid:
                        x, y, w, h = ann['bbox']
                        (x1, y1), (x2, y2) = mot17_to_qwen_bbox(x, y, w, h, img_height, img_width)
                        frame1_lines.append(
                            f"<|object_ref_start|>person<|object_ref_end|><|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"
                        )
                        break
            response_parts.append("\n".join(frame1_lines))

            # Frame 2 检测 (按相同sorted_ids顺序)
            frame2_lines = ["Frame 2:"]
            for tid in sorted_ids:
                for ann in sample_frames[1]['annotations']:
                    if ann['id'] == tid:
                        x, y, w, h = ann['bbox']
                        (x1, y1), (x2, y2) = mot17_to_qwen_bbox(x, y, w, h, img_height, img_width)
                        frame2_lines.append(
                            f"<|object_ref_start|>person<|object_ref_end|><|box_start|>({x1},{y1}),({x2},{y2})<|box_end|>"
                        )
                        break
            response_parts.append("\n".join(frame2_lines))

            # Matching: 0-0, 1-1, 2-2, ... (因为按相同顺序排列)
            matching_pairs = [f"{i}-{i}" for i in range(len(sorted_ids))]
            response_parts.append(f"Matching: {', '.join(matching_pairs)}")

            assistant_content = [
                {"type": "text", "text": "\n\n".join(response_parts)}
            ]

            sample = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "metadata": {
                    "source": seq_name,
                    "frame_ids": [f['frame_id'] for f in sample_frames],
                    "num_tracks": len(sorted_ids),
                    "stage": 2,
                    "format": "detect_match",
                }
            }
            samples.append(sample)

    return samples


# ============================================================
# 数据保存与统计
# ============================================================

def save_jsonl(samples, output_path):
    """保存为 JSONL 格式"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"  保存: {output_path} ({len(samples)} 样本)")


def print_statistics(samples, stage_name):
    """打印数据集统计信息"""
    if not samples:
        print(f"\n  {stage_name}: 无样本")
        return

    num_samples = len(samples)
    num_objects = [s['metadata']['num_objects'] if 'num_objects' in s['metadata']
                   else s['metadata']['num_tracks']
                   for s in samples]
    sources = [s['metadata']['source'] for s in samples]

    print(f"\n  {stage_name} 统计:")
    print(f"    样本数: {num_samples}")
    print(f"    平均目标数: {sum(num_objects) / len(num_objects):.1f}")
    print(f"    最小目标数: {min(num_objects)}")
    print(f"    最大目标数: {max(num_objects)}")
    print(f"    序列分布: {dict(sorted(defaultdict(int, ((s, sources.count(s)) for s in set(sources))).items()))}")


# ============================================================
# 数据质量验证
# ============================================================

def validate_sample(sample, stage):
    """验证单个样本的格式正确性"""
    errors = []

    # 检查 messages 结构
    if 'messages' not in sample:
        errors.append("缺少 messages 字段")
        return errors

    messages = sample['messages']
    if len(messages) != 2:
        errors.append(f"messages 应有 2 条, 实际 {len(messages)} 条")
        return errors

    user_msg, assistant_msg = messages[0], messages[1]

    # 检查角色
    if user_msg.get('role') != 'user':
        errors.append(f"第一条消息角色应为 user, 实际 {user_msg.get('role')}")
    if assistant_msg.get('role') != 'assistant':
        errors.append(f"第二条消息角色应为 assistant, 实际 {assistant_msg.get('role')}")

    # 检查用户消息内容
    user_content = user_msg.get('content', [])
    has_image = any(c.get('type') == 'image' for c in user_content)
    has_text = any(c.get('type') == 'text' for c in user_content)
    if not has_image:
        errors.append("用户消息缺少图像")
    if not has_text:
        errors.append("用户消息缺少文本提示词")

    # 检查助手回复
    assistant_text = ""
    for c in assistant_msg.get('content', []):
        if c.get('type') == 'text':
            assistant_text += c.get('text', '')

    if not assistant_text:
        errors.append("助手回复为空")
    elif '<|box_start|>' not in assistant_text:
        errors.append("助手回复缺少 <|box_start|> 标签")

    # 检查坐标范围 (0-1000)
    import re
    box_pattern = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    boxes = re.findall(box_pattern, assistant_text)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if not (0 <= x1 <= 999 and 0 <= y1 <= 999 and 0 < x2 <= 1000 and 0 < y2 <= 1000):
            errors.append(f"bbox {i} 坐标超出范围: ({x1},{y1}),({x2},{y2})")
        if x2 <= x1 or y2 <= y1:
            errors.append(f"bbox {i} 无效: x2<=x1 或 y2<=y1")

    return errors


def validate_dataset(samples, stage):
    """验证整个数据集"""
    print(f"\n  验证 Stage {stage} 数据集...")
    error_count = 0
    error_types = defaultdict(int)

    for i, sample in enumerate(samples):
        errors = validate_sample(sample, stage)
        if errors:
            error_count += 1
            for e in errors:
                error_types[e] += 1
            if error_count <= 5:  # 只打印前5个错误样本
                print(f"    样本 {i} 错误: {errors}")

    if error_count == 0:
        print(f"  ✅ 所有 {len(samples)} 个样本验证通过")
    else:
        print(f"  ⚠️ {error_count}/{len(samples)} 个样本有错误")
        for etype, cnt in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"    - {etype}: {cnt} 次")

    return error_count == 0


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MOT17 → Qwen2-VL SFT 数据构建")
    parser.add_argument(
        '--mot17-root', type=str, default=str(MOT17_ROOT),
        help='MOT17 数据集根目录'
    )
    parser.add_argument(
        '--output-dir', type=str, default=str(OUTPUT_DIR),
        help='输出目录'
    )
    parser.add_argument(
        '--stages', type=str, default='1,2',
        help='要构建的阶段 (逗号分隔, 如 1,2)'
    )
    parser.add_argument(
        '--skip-validation', action='store_true',
        help='跳过数据验证'
    )
    args = parser.parse_args()

    mot17_root = Path(args.mot17_root)
    output_dir = Path(args.output_dir)
    stages = [int(s.strip()) for s in args.stages.split(',')]

    print("=" * 60)
    print("Phase 2: MOT17 → Qwen2-VL SFT 数据构建")
    print("=" * 60)
    print(f"  MOT17 路径: {mot17_root}")
    print(f"  输出目录: {output_dir}")
    print(f"  构建阶段: {stages}")
    print()

    if not mot17_root.exists():
        print(f"❌ MOT17 路径不存在: {mot17_root}")
        sys.exit(1)

    # 统计可用序列
    seq_names = [d for d in sorted(os.listdir(mot17_root))
                 if (mot17_root / d).is_dir() and (mot17_root / d / "gt" / "gt.txt").exists()]
    print(f"  可用序列: {seq_names}")
    print()

    # Stage 1: 单帧检测数据 (原生 Qwen2-VL 格式, 利用模型已有检测能力)
    if 1 in stages:
        print("-" * 60)
        print("构建 Stage 1: 单帧检测数据 (原生格式)")
        print("-" * 60)
        # stride=5 采样更密, 单帧检测无跨帧约束, 样本量更大
        stage1_samples = build_stage1_samples(mot17_root, stride=5)
        print_statistics(stage1_samples, "Stage 1 (单帧检测)")

        if not args.skip_validation:
            validate_dataset(stage1_samples, 1)

        save_jsonl(stage1_samples, output_dir / "mot17_sft_stage1.jsonl")

    # Stage 2: 双帧跟踪数据 (用于后续 ID 跟踪格式学习或 RL 精调)
    if 2 in stages:
        print("\n" + "-" * 60)
        print("构建 Stage 2: 双帧跟踪数据")
        print("-" * 60)
        stage2_samples = build_stage2_samples(mot17_root, stride=10)
        print_statistics(stage2_samples, "Stage 2")

        if not args.skip_validation:
            validate_dataset(stage2_samples, 2)

        save_jsonl(stage2_samples, output_dir / "mot17_sft_stage2.jsonl")

    # 总结
    print("\n" + "=" * 60)
    print("Phase 2 数据构建完成")
    print("=" * 60)
    print(f"""
  生成的数据文件:
    Stage 1 (单帧检测, 原生格式):           {output_dir / 'mot17_sft_stage1.jsonl'}
    Stage 2 (双帧跟踪, 用于后续RL/ID学习):  {output_dir / 'mot17_sft_stage2.jsonl'}

  下一步: Stage 1 单帧检测 SFT 训练
  运行: python model_method/train.py --stage 1
    """)


if __name__ == '__main__':
    main()
