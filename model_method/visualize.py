"""
Phase 3.5: 训练前模型推理可视化

在 LoRA 微调之前，测试 Qwen2-VL 原始模型在 MOT17 上的行人检测与跟踪能力。
帮助区分: 模型本身能力不足 vs 微调训练不到位。

使用方式:
  python model_method/visualize.py --mode detect
  python model_method/visualize.py --mode track
  python model_method/visualize.py --mode track10
  python model_method/visualize.py --mode compare --lora-path runs/stage1/final
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "Qwen"
MOT17_PATH = PROJECT_ROOT / "data" / "MOT17"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "visualizations"

# 评估参数 (由 main() 设置)
EVAL_SEQ_FILTER = None  # 只测试指定序列, 如 ['MOT17-02-FRCNN', 'MOT17-11-FRCNN']
EVAL_MAX_FRAMES = None  # 每个序列最多测试的帧数

# 修复 Windows GBK 编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# bbox 解析与绘制
# ============================================================

def parse_bboxes_from_text(text):
    """
    从模型输出文本中解析 bbox

    支持两种格式:
    1. <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    2. <box>(x1,y1),(x2,y2)</box>
    """
    bboxes = []

    # 格式1: 特殊 token 格式
    pattern1 = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    for m in re.finditer(pattern1, text):
        bboxes.append((int(m.group(1)), int(m.group(2)),
                       int(m.group(3)), int(m.group(4))))

    # 格式2: 文本标签格式
    pattern2 = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    for m in re.finditer(pattern2, text):
        bboxes.append((int(m.group(1)), int(m.group(2)),
                       int(m.group(3)), int(m.group(4))))

    return bboxes


def parse_tracks_from_text(text):
    """
    从模型输出文本中解析跟踪结果 (带 ID)

    支持多种格式变体:
    - ID1: <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
    - - ID1: ... (带前缀)
    - ID1: ..., ID2: ... (多个ID在同一行)
    - ID0: ... at (x,y), ID1: ... (带额外文本)
    """
    tracks = {}

    # 用正则找到所有 "ID数字:" 模式, 将文本分割为每个ID对应的片段
    # 匹配: ID数字: (可能有前缀如 "- ", 逗号等)
    id_pattern = r'(?:^|[\s,])-?\s*ID(\d+)\s*:'

    # 找到所有ID位置
    id_matches = list(re.finditer(id_pattern, text))

    for idx, match in enumerate(id_matches):
        track_id = int(match.group(1))

        # 提取该ID对应的文本片段: 从当前ID到下一个ID(或行尾)
        start = match.end()
        if idx + 1 < len(id_matches):
            end = id_matches[idx + 1].start()
        else:
            # 最后一个ID, 取到行尾或文本末尾
            remaining = text[start:]
            # 取到下一个换行或文本末尾
            newline_pos = remaining.find('\n')
            end = start + (newline_pos if newline_pos >= 0 else len(remaining))

        segment = text[start:end]

        # 从片段中提取 bbox
        bboxes = parse_bboxes_from_text(segment)
        if bboxes:
            if track_id not in tracks:
                tracks[track_id] = []
            # 只取第一个bbox (该ID在该帧的bbox)
            tracks[track_id].append(bboxes[0])

    return tracks


def parse_detect_match_response(response):
    """
    解析"先检测再匹配"格式的输出

    格式:
      Frame 1:
      <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
      ...

      Frame 2:
      <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
      ...

      Matching: 0-0, 1-1, 2-2

    Returns:
        frame1_bboxes: [(x1,y1,x2,y2), ...]  # Qwen 0-1000坐标
        frame2_bboxes: [(x1,y1,x2,y2), ...]
        matching: [(i, j), ...]  # Frame1[i] <-> Frame2[j]
    """
    # 按 "Frame 1:" 和 "Frame 2:" 分割
    frame1_bboxes = []
    frame2_bboxes = []
    matching = []

    # 提取 Frame 1 部分
    frame1_match = re.search(r'Frame\s*1\s*:(.*?)(?=Frame\s*2\s*:|Matching\s*:|$)', response, re.DOTALL)
    if frame1_match:
        frame1_bboxes = parse_bboxes_from_text(frame1_match.group(1))

    # 提取 Frame 2 部分
    frame2_match = re.search(r'Frame\s*2\s*:(.*?)(?=Matching\s*:|$)', response, re.DOTALL)
    if frame2_match:
        frame2_bboxes = parse_bboxes_from_text(frame2_match.group(1))

    # 提取 Matching 部分
    matching_match = re.search(r'Matching\s*:\s*(.*?)(?:\n|$)', response, re.DOTALL)
    if matching_match:
        matching_str = matching_match.group(1).strip()
        # 解析 "0-0, 1-1, 2-2" 格式
        pairs = re.findall(r'(\d+)\s*-\s*(\d+)', matching_str)
        for i, j in pairs:
            matching.append((int(i), int(j)))

    return frame1_bboxes, frame2_bboxes, matching


def qwen_bbox_to_pixel(x1, y1, x2, y2, img_width, img_height):
    """将 Qwen2-VL 0-1000 坐标转换回像素坐标"""
    px1 = int(x1 / 1000 * img_width)
    py1 = int(y1 / 1000 * img_height)
    px2 = int(x2 / 1000 * img_width)
    py2 = int(y2 / 1000 * img_height)
    return px1, py1, px2, py2


def draw_bboxes_on_image(image, bboxes, color='red', label=None, img_width=None, img_height=None):
    """在图像上绘制 bbox"""
    draw = ImageDraw.Draw(image)

    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # 转换坐标
        if img_width and img_height:
            px1, py1, px2, py2 = qwen_bbox_to_pixel(x1, y1, x2, y2, img_width, img_height)
        else:
            px1, py1, px2, py2 = x1, y1, x2, y2

        # 绘制矩形
        draw.rectangle([px1, py1, px2, py2], outline=color, width=2)

        # 绘制标签
        text = label if label else f"{i+1}"
        draw.text((px1, py1 - 18), text, fill=color, font=font)

    return image


def draw_tracks_on_image(image, tracks, img_width, img_height):
    """在图像上绘制跟踪结果 (带 ID 的彩色 bbox)"""
    import colorsys

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # 为每个 ID 生成不同颜色
    colors = {}
    for track_id in tracks:
        hue = (track_id * 0.618) % 1.0  # 黄金比例分散
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors[track_id] = (int(r * 255), int(g * 255), int(b * 255))

    for track_id, bboxes in tracks.items():
        color = colors[track_id]
        for x1, y1, x2, y2 in bboxes:
            px1, py1, px2, py2 = qwen_bbox_to_pixel(x1, y1, x2, y2, img_width, img_height)
            draw.rectangle([px1, py1, px2, py2], outline=color, width=2)
            draw.text((px1, py1 - 18), f"ID{track_id}", fill=color, font=font)

    return image


# ============================================================
# 推理函数
# ============================================================

def load_model(model_path, lora_path=None):
    """加载模型 (可选加载 LoRA 权重)"""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    print(f"  加载模型: {model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(str(model_path))

    # 加载 LoRA 权重
    if lora_path:
        from peft import PeftModel
        print(f"  加载 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    return model, processor


def run_detection(model, processor, image_path, prompt=None):
    """单帧行人检测"""
    from qwen_vl_utils import process_vision_info

    if prompt is None:
        prompt = (
            "Detect all pedestrians in this image. "
            "For each person, output their bounding box. "
            "Use format: <|object_ref_start|>person<|object_ref_end|>"
            "<|box_start|>(x1,y1),(x2,y2)<|box_end|>"
        )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,  # 降低到1024, 双帧约45个行人足够
            do_sample=False,
        )

    # 只取生成的部分
    generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    return response


def run_tracking(model, processor, image_paths, prompt=None):
    """多帧行人跟踪 (一次性传入所有帧)"""
    from qwen_vl_utils import process_vision_info

    num_frames = len(image_paths)

    if prompt is None:
        prompt = (
            f"Track all pedestrians across these {num_frames} frames. "
            "For each person, assign a unique tracking ID and provide their "
            "bounding box in each frame. "
            "Output format per frame: Frame N: IDx: "
            "<|object_ref_start|>person<|object_ref_end|>"
            "<|box_start|>(x1,y1),(x2,y2)<|box_end|> "
            "Coordinates are normalized to 0-1000."
        )

    # 构建消息
    content = []
    for path in image_paths:
        content.append({"type": "image", "image": path})
    content.append({"type": "text", "text": prompt})

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,  # 降低到1024, 双帧约45个行人足够
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    return response


def run_tracking_sliding(model, processor, image_paths):
    """
    滑动窗口跟踪: 每次传入2帧, 与Stage 2训练方式完全一致

    流程:
    1. [frame0, frame1] + 训练prompt → 输出 Frame 1 + Frame 2
    2. [frame1, frame2] + 训练prompt → 输出 Frame 1 + Frame 2
    3. ...

    每帧取最新结果: frame0用Step0的Frame1, frame_i用Step_i的Frame1
    """
    from qwen_vl_utils import process_vision_info

    num_frames = len(image_paths)
    frame_results = [None] * num_frames  # 每帧的预测结果

    # 训练数据中的prompt (完全匹配Stage 2训练格式)
    prompt_track = (
        "Track all pedestrians across these 2 consecutive frames. "
        "For each person, assign a unique tracking ID and provide their bounding box in each frame. "
        "Output format per frame: Frame N: IDx: "
        "<|object_ref_start|>person<|object_ref_end|>"
        "<|box_start|>(x1,y1),(x2,y2)<|box_end|> "
        "Coordinates are normalized to 0-1000."
    )

    def extract_frame(response, frame_num):
        """从模型输出中提取指定帧的结果"""
        # 匹配 "Frame N:" 到下一个 "Frame" 或末尾
        import re
        pattern = rf"Frame {frame_num}:\s*(.*?)(?=Frame \d+:|$)"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response  # 如果没匹配到, 返回全部

    # 滑动窗口: 每次处理2帧
    num_steps = max(1, num_frames - 1)
    for step_idx in tqdm(range(num_steps), desc="    跟踪", unit="步"):
        i = step_idx
        j = min(i + 1, num_frames - 1)

        content = [
            {"type": "image", "image": image_paths[i]},
            {"type": "image", "image": image_paths[j]},
            {"type": "text", "text": prompt_track},
        ]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        # 提取两帧的结果
        frame_results[i] = extract_frame(response, 1)  # Frame 1 = 第i帧
        if j > i:
            frame_results[j] = extract_frame(response, 2)  # Frame 2 = 第j帧

    # 合并所有帧的响应
    combined = "\n".join(
        f"Frame {i+1}: {resp or 'No detection'}"
        for i, resp in enumerate(frame_results)
    )
    return combined


def run_detect_match_tracking(model, processor, image_paths):
    """
    先检测再匹配的滑动窗口推理

    流程:
    1. 每次传入2帧, 用detect-match prompt
    2. 解析: Frame1检测框 + Frame2检测框 + Matching关系
    3. 根据Matching关联两帧的检测框, 分配track ID
    4. 跨窗口: 用IoU匹配传递ID

    Returns:
        all_frame_bboxes: List[List[(x1,y1,x2,y2)]]  # 每帧的bbox列表 (像素坐标)
        all_frame_ids: List[List[int]]               # 每帧的track ID列表
    """
    from qwen_vl_utils import process_vision_info

    num_frames = len(image_paths)

    # detect-match prompt (与训练数据格式一致)
    prompt_dm = (
        "Detect all pedestrians in these 2 consecutive frames, then match them across frames.\n\n"
        "Step 1 - Detection: For each frame, output all pedestrian bounding boxes (one per line).\n"
        "Format: <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>\n\n"
        "Step 2 - Matching: Output the correspondence between Frame 1 and Frame 2 detections.\n"
        "Format: Matching: i-j (where i is the index in Frame 1, j is the index in Frame 2)\n\n"
        "Coordinates are normalized to 0-1000."
    )

    # 每帧的检测结果 (Qwen 0-1000坐标)
    frame_detections = [None] * num_frames
    # 跨窗口的track ID
    frame_ids = [None] * num_frames

    # 滑动窗口: 每次处理2帧
    num_steps = max(1, num_frames - 1)
    next_global_id = 1
    prev_bboxes = None  # 上一窗口最后一帧的bbox (像素坐标)
    prev_ids = None     # 上一窗口最后一帧的ID

    for step_idx in tqdm(range(num_steps), desc="    Detect-Match", unit="步"):
        i = step_idx
        j = min(i + 1, num_frames - 1)

        content = [
            {"type": "image", "image": image_paths[i]},
            {"type": "image", "image": image_paths[j]},
            {"type": "text", "text": prompt_dm},
        ]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        # 注意: 必须保留特殊 token (<|box_start|> 等), 否则解析器无法匹配 bbox
        response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

        if step_idx == 0:
            print(f"    模型输出 (Step 0): {response[:500]}...")

        # 解析detect-match格式
        f1_bboxes, f2_bboxes, matching = parse_detect_match_response(response)
        print(f"    Step {step_idx}: Frame1={len(f1_bboxes)}框, Frame2={len(f2_bboxes)}框, Matching={len(matching)}对")

        # 转换为像素坐标
        # 注意: 这里需要图像尺寸, 暂时用Qwen坐标, 后续在可视化时转换
        f1_qwen = f1_bboxes  # [(x1,y1,x2,y2), ...] Qwen 0-1000
        f2_qwen = f2_bboxes

        # 为当前窗口分配ID
        if step_idx == 0:
            # 第一个窗口: Frame1的所有检测框分配新ID
            f1_ids = list(range(next_global_id, next_global_id + len(f1_qwen)))
            next_global_id += len(f1_qwen)

            # Frame2: 根据matching分配ID
            f2_ids = [None] * len(f2_qwen)
            for fi, fj in matching:
                if fi < len(f1_ids) and fj < len(f2_ids):
                    f2_ids[fj] = f1_ids[fi]
            # 未匹配的分配新ID
            for k in range(len(f2_ids)):
                if f2_ids[k] is None:
                    f2_ids[k] = next_global_id
                    next_global_id += 1

            frame_detections[i] = f1_qwen
            frame_detections[j] = f2_qwen
            frame_ids[i] = f1_ids
            frame_ids[j] = f2_ids

            prev_bboxes = f2_qwen
            prev_ids = f2_ids
        else:
            # 后续窗口: Frame1 = 上一窗口的Frame2
            # 用IoU匹配上一窗口的Frame2和当前窗口的Frame1
            f1_ids = [None] * len(f1_qwen)
            if prev_bboxes and prev_ids:
                for k, bbox in enumerate(f1_qwen):
                    best_iou = 0.0
                    best_id = None
                    for pi, pbbox in enumerate(prev_bboxes):
                        iou = compute_iou(bbox, pbbox)
                        if iou > best_iou and iou > 0.3:
                            best_iou = iou
                            best_id = prev_ids[pi]
                    if best_id is not None:
                        f1_ids[k] = best_id
                    else:
                        f1_ids[k] = next_global_id
                        next_global_id += 1
            else:
                f1_ids = list(range(next_global_id, next_global_id + len(f1_qwen)))
                next_global_id += len(f1_qwen)

            # Frame2: 根据matching分配ID
            f2_ids = [None] * len(f2_qwen)
            for fi, fj in matching:
                if fi < len(f1_ids) and fj < len(f2_ids):
                    f2_ids[fj] = f1_ids[fi]
            for k in range(len(f2_ids)):
                if f2_ids[k] is None:
                    f2_ids[k] = next_global_id
                    next_global_id += 1

            frame_detections[j] = f2_qwen
            frame_ids[j] = f2_ids

            prev_bboxes = f2_qwen
            prev_ids = f2_ids

    return frame_detections, frame_ids


def run_detection_per_frame(model, processor, image_path, prompt=None):
    """
    单帧行人检测 (原生 Qwen2-VL 格式)

    与训练数据 Stage 1 格式完全一致:
      输入: 单帧图像 + 检测提示词
      输出: <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>
    """
    from qwen_vl_utils import process_vision_info

    if prompt is None:
        prompt = (
            "Detect all pedestrians in this image. "
            "For each person, output their bounding box. "
            "Use format: <|object_ref_start|>person<|object_ref_end|>"
            "<|box_start|>(x1,y1),(x2,y2)<|box_end|>"
        )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
    # 保留特殊 token, 解析器依赖 <|box_start|> 等标记
    response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    return response


def detect_and_track_iou(model, processor, image_paths, iou_threshold=0.3):
    """
    检测 + IoU 后处理跟踪 (方案 B Stage 2 核心)

    流程:
    1. 对每帧独立检测行人 (原生 Qwen2-VL 格式, 利用模型已有能力)
    2. 跨帧 IoU 匹配关联轨迹 (简化版 SORT, 无卡尔曼滤波)
       - IoU > threshold: 保持同一 track ID
       - 无匹配: 分配新 track ID

    Returns:
        frame_detections: List[List[(x1,y1,x2,y2)]]  # 每帧的bbox (Qwen 0-1000坐标)
        frame_ids: List[List[int]]                    # 每帧的track ID
    """
    num_frames = len(image_paths)
    frame_detections = [None] * num_frames
    frame_ids = [None] * num_frames

    next_global_id = 1
    prev_bboxes = None  # 上一帧的 bbox (Qwen 0-1000坐标)
    prev_ids = None     # 上一帧的 track ID

    for step_idx in tqdm(range(num_frames), desc="    Detect-Track", unit="帧"):
        # 1. 单帧检测
        response = run_detection_per_frame(model, processor, image_paths[step_idx])

        if step_idx == 0:
            print(f"    模型输出 (Frame 0): {response[:300]}...")

        # 2. 解析 bbox
        bboxes = parse_bboxes_from_text(response)

        # 3. NMS 去除重叠框
        bboxes = nms(bboxes, iou_threshold=0.5)

        # 4. IoU 匹配分配 track ID
        if step_idx == 0:
            # 第一帧: 所有检测分配新 ID
            ids = list(range(next_global_id, next_global_id + len(bboxes)))
            next_global_id += len(bboxes)
        else:
            # 后续帧: 与上一帧 IoU 匹配
            ids = [None] * len(bboxes)
            if prev_bboxes and prev_ids:
                # 贪心 IoU 匹配
                matched_prev = set()
                for k, bbox in enumerate(bboxes):
                    best_iou = 0.0
                    best_prev_idx = -1
                    for pi, pbbox in enumerate(prev_bboxes):
                        if pi in matched_prev:
                            continue
                        iou = compute_iou(bbox, pbbox)
                        if iou > best_iou and iou > iou_threshold:
                            best_iou = iou
                            best_prev_idx = pi
                    if best_prev_idx >= 0:
                        ids[k] = prev_ids[best_prev_idx]
                        matched_prev.add(best_prev_idx)
            # 未匹配的分配新 ID
            for k in range(len(ids)):
                if ids[k] is None:
                    ids[k] = next_global_id
                    next_global_id += 1

        frame_detections[step_idx] = bboxes
        frame_ids[step_idx] = ids

        prev_bboxes = bboxes
        prev_ids = ids

        print(f"    Frame {step_idx}: 检测={len(bboxes)}框, 累计ID数={next_global_id-1}")

    return frame_detections, frame_ids


# ============================================================
# 可视化主函数
# ============================================================

def compute_iou(box1, box2):
    """计算两个bbox的IoU (格式: x1,y1,x2,y2)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter

    return inter / max(union, 1e-8)


def nms(bboxes, iou_threshold=0.5):
    """
    非极大值抑制 (NMS)

    去除重叠度过高的重复检测框, 保留面积最大的框。
    解决密集人群区域同一人被多次检测的问题。

    Args:
        bboxes: List[(x1,y1,x2,y2)] 预测bbox列表
        iou_threshold: IoU超过此阈值时抑制较小的框

    Returns:
        List[(x1,y1,x2,y2)] 过滤后的bbox列表
    """
    if len(bboxes) <= 1:
        return bboxes

    # 按面积从大到小排序 (保留更大的框, 更可能是完整的人)
    sorted_boxes = sorted(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)

    keep = []
    for box in sorted_boxes:
        should_keep = True
        for kept_box in keep:
            if compute_iou(box, kept_box) > iou_threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(box)

    return keep


def match_predictions_to_gt(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    """
    将预测bbox与GT bbox进行匹配 (贪心算法)

    Returns:
        tp: True Positives (正确检测数)
        fp: False Positives (误检数)
        fn: False Negatives (漏检数)
    """
    if not pred_bboxes or not gt_bboxes:
        return 0, len(pred_bboxes), len(gt_bboxes)

    # 计算所有IoU
    iou_matrix = []
    for i, pred in enumerate(pred_bboxes):
        row = []
        for j, gt in enumerate(gt_bboxes):
            row.append(compute_iou(pred, gt))
        iou_matrix.append(row)

    # 贪心匹配
    matched_gt = set()
    matched_pred = set()
    tp = 0

    # 按IoU从高到低匹配
    while True:
        max_iou = 0
        max_i, max_j = -1, -1

        for i in range(len(pred_bboxes)):
            if i in matched_pred:
                continue
            for j in range(len(gt_bboxes)):
                if j in matched_gt:
                    continue
                if iou_matrix[i][j] > max_iou:
                    max_iou = iou_matrix[i][j]
                    max_i, max_j = i, j

        if max_iou < iou_threshold or max_i == -1:
            break

        matched_pred.add(max_i)
        matched_gt.add(max_j)
        tp += 1

    fp = len(pred_bboxes) - tp  # 未匹配的预测
    fn = len(gt_bboxes) - tp    # 未匹配的GT

    return tp, fp, fn


def visualize_detection(model, processor, output_dir):
    """可视化单帧检测"""
    print("\n" + "=" * 60)
    print("单帧行人检测可视化")
    print("=" * 60)

    output_dir = Path(output_dir) / "detect"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 选择测试序列
    seq_names = sorted([d for d in os.listdir(MOT17_PATH)
                        if (MOT17_PATH / d).is_dir()
                        and (MOT17_PATH / d / "gt" / "gt.txt").exists()])

    # 序列过滤
    if EVAL_SEQ_FILTER:
        seq_names = [s for s in seq_names if s in EVAL_SEQ_FILTER]

    total_detected = 0
    total_gt = 0
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives
    results = []

    for seq_name in seq_names[:3]:  # 只测试前3个序列
        seq_path = MOT17_PATH / seq_name
        img_dir = seq_path / "img1"
        gt_path = seq_path / "gt" / "gt.txt"

        # 读取序列信息
        info = {}
        seqinfo_path = seq_path / "seqinfo.ini"
        if seqinfo_path.exists():
            with open(seqinfo_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, val = line.split('=', 1)
                        info[key.strip()] = val.strip()
        img_width = int(info.get('imWidth', 1920))
        img_height = int(info.get('imHeight', 1080))

        # 解析 GT
        from model_method.build_sft_data import parse_gt
        annotations = parse_gt(gt_path)

        # 测试3帧
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        test_frames = [frames[0], frames[len(frames)//2], frames[-1]]

        for frame_file in tqdm(test_frames, desc=f"  {seq_name} 检测", unit="帧"):
            frame_id = int(frame_file.replace('.jpg', ''))
            image_path = str(img_dir / frame_file)

            # GT
            gt_anns = annotations.get(frame_id, [])
            gt_bboxes = []
            for ann in gt_anns:
                x, y, w, h = ann['bbox']
                from model_method.build_sft_data import mot17_to_qwen_bbox
                (x1, y1), (x2, y2) = mot17_to_qwen_bbox(x, y, w, h, img_height, img_width)
                gt_bboxes.append((x1, y1, x2, y2))

            # 模型推理
            response = run_detection(model, processor, image_path)

            # 解析预测
            pred_bboxes = parse_bboxes_from_text(response)
            pred_bboxes_raw = pred_bboxes
            # NMS: 去除重叠框 (密集人群区域同一人可能被多次检测)
            pred_bboxes = nms(pred_bboxes, iou_threshold=0.5)

            # 计算TP, FP, FN (IoU阈值=0.5)
            tp, fp, fn = match_predictions_to_gt(pred_bboxes, gt_bboxes, iou_threshold=0.5)

            print(f"    GT 行人数: {len(gt_bboxes)}")
            print(f"    预测行人数: {len(pred_bboxes_raw)} → NMS后: {len(pred_bboxes)}")
            print(f"    TP={tp}, FP={fp}, FN={fn}")
            print(f"    模型输出: {response[:200]}...")

            # 绘制可视化
            image = Image.open(image_path).convert("RGB")

            # 绘制 GT (绿色)
            gt_image = image.copy()
            draw_bboxes_on_image(gt_image, gt_bboxes, color='green', label='GT',
                                img_width=img_width, img_height=img_height)

            # 绘制预测 (红色)
            pred_image = image.copy()
            draw_bboxes_on_image(pred_image, pred_bboxes, color='red', label='Pred',
                                img_width=img_width, img_height=img_height)

            # 合并图
            combined = Image.new('RGB', (img_width * 2 + 20, img_height + 40), 'white')
            combined.paste(gt_image, (0, 0))
            combined.paste(pred_image, (img_width + 20, 0))

            # 添加标题
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            draw.text((10, img_height + 5), "Ground Truth (Green)", fill='green', font=font)
            draw.text((img_width + 30, img_height + 5), "Prediction (Red)", fill='red', font=font)

            # 保存
            save_name = f"{seq_name}_{frame_file}"
            combined.save(str(output_dir / save_name))
            print(f"    保存: {output_dir / save_name}")

            total_detected += len(pred_bboxes)
            total_gt += len(gt_bboxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            results.append({
                'sequence': seq_name,
                'frame': frame_file,
                'gt_count': len(gt_bboxes),
                'pred_count': len(pred_bboxes),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'response_preview': response[:200],
            })

    # 总结
    print(f"\n{'=' * 60}")
    print(f"检测总结 (IoU阈值=0.5)")
    print(f"{'=' * 60}")
    print(f"  总 GT 行人数: {total_gt}")
    print(f"  总预测行人数: {total_detected}")
    print(f"  True Positives (TP): {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")

    # 计算指标
    precision = total_tp / max(total_tp + total_fp, 1) * 100
    recall = total_tp / max(total_tp + total_fn, 1) * 100
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n  精确率 (Precision): {precision:.1f}%")
    print(f"  召回率 (Recall): {recall:.1f}%")
    print(f"  F1 分数: {f1:.1f}%")

    # 保存结果
    with open(str(output_dir / "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def visualize_tracking(model, processor, output_dir):
    """可视化多帧跟踪"""
    print("\n" + "=" * 60)
    print("多帧行人跟踪可视化")
    print("=" * 60)

    output_dir = Path(output_dir) / "track"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 选择测试序列
    seq_names = sorted([d for d in os.listdir(MOT17_PATH)
                        if (MOT17_PATH / d).is_dir()
                        and (MOT17_PATH / d / "gt" / "gt.txt").exists()])

    for seq_name in seq_names[:2]:  # 只测试前2个序列
        seq_path = MOT17_PATH / seq_name
        img_dir = seq_path / "img1"

        info = {}
        seqinfo_path = seq_path / "seqinfo.ini"
        if seqinfo_path.exists():
            with open(seqinfo_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, val = line.split('=', 1)
                        info[key.strip()] = val.strip()
        img_width = int(info.get('imWidth', 1920))
        img_height = int(info.get('imHeight', 1080))

        frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

        # 选择4帧 (间隔5帧)
        frame_indices = [0, 5, 10, 15]
        image_paths = [str(img_dir / frames[i]) for i in frame_indices if i < len(frames)]

        if len(image_paths) < 2:
            continue

        print(f"\n  {seq_name}: {len(image_paths)} 帧")

        # 模型推理
        print(f"    推理中 ({len(image_paths)} 帧)...")
        response = run_tracking(model, processor, image_paths)

        print(f"    模型输出: {response[:300]}...")

        # 解析跟踪结果
        tracks = parse_tracks_from_text(response)
        print(f"    检测到 track ID 数: {len(tracks)}")

        # 为每帧绘制可视化
        for i, image_path in enumerate(tqdm(image_paths, desc="    绘制", unit="帧")):
            image = Image.open(image_path).convert("RGB")

            # 绘制跟踪结果
            if tracks:
                draw_tracks_on_image(image, tracks, img_width, img_height)

            # 保存
            frame_name = Path(image_path).stem
            save_name = f"{seq_name}_frame{i+1}_{frame_name}.jpg"
            image.save(str(output_dir / save_name))
            print(f"    保存: {output_dir / save_name}")


def visualize_track10(model, processor, output_dir):
    """可视化10帧连续跟踪 (GT vs Pred 对比)"""
    print("\n" + "=" * 60)
    print("10帧连续跟踪可视化 (GT vs Pred)")
    print("=" * 60)

    output_dir = Path(output_dir) / "track10"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 选择测试序列
    seq_names = sorted([d for d in os.listdir(MOT17_PATH)
                        if (MOT17_PATH / d).is_dir()
                        and (MOT17_PATH / d / "gt" / "gt.txt").exists()])

    # 序列过滤
    if EVAL_SEQ_FILTER:
        seq_names = [s for s in seq_names if s in EVAL_SEQ_FILTER]

    from model_method.build_sft_data import parse_gt, mot17_to_qwen_bbox

    for seq_name in seq_names[:2]:  # 只测试前2个序列
        seq_path = MOT17_PATH / seq_name
        img_dir = seq_path / "img1"
        gt_path = seq_path / "gt" / "gt.txt"

        # 读取序列信息
        info = {}
        seqinfo_path = seq_path / "seqinfo.ini"
        if seqinfo_path.exists():
            with open(seqinfo_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, val = line.split('=', 1)
                        info[key.strip()] = val.strip()
        img_width = int(info.get('imWidth', 1920))
        img_height = int(info.get('imHeight', 1080))

        # 解析 GT
        annotations = parse_gt(gt_path)

        # 选择连续帧
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        _default_n = 10
        _max_n = EVAL_MAX_FRAMES if EVAL_MAX_FRAMES else _default_n
        num_frames = min(_max_n, len(frames))
        frame_indices = list(range(num_frames))
        image_paths = [str(img_dir / frames[i]) for i in frame_indices]

        if len(image_paths) < 2:
            continue

        print(f"\n  {seq_name}: {len(image_paths)} 连续帧 (帧1-{num_frames})")

        # 模型推理 (滑动窗口: 每次2帧, 带文本记忆)
        print(f"    滑动窗口推理中...")
        response = run_tracking_sliding(model, processor, image_paths)

        print(f"    模型输出: {response[:500]}...")

        # 解析跟踪结果
        tracks = parse_tracks_from_text(response)
        print(f"    检测到 track ID 数: {len(tracks)}")

        # 为每帧构建 GT 和 Pred 的 bbox 列表
        all_frame_data = []
        total_iou = 0.0
        total_iou_count = 0
        total_id_match = 0
        total_id_count = 0

        for i, image_path in enumerate(tqdm(image_paths, desc="    解析", unit="帧")):
            frame_file = frames[frame_indices[i]]
            frame_id = int(frame_file.replace('.jpg', ''))

            # GT bboxes (像素坐标)
            gt_anns = annotations.get(frame_id, [])
            gt_bboxes_pixel = []
            gt_ids = []
            for ann in gt_anns:
                x, y, w, h = ann['bbox']
                gt_bboxes_pixel.append((int(x), int(y), int(x + w), int(y + h)))
                gt_ids.append(ann.get('id', -1))

            # Pred bboxes (从 qwen 0-1000 坐标转像素坐标)
            pred_bboxes_pixel = []
            pred_ids = []
            for track_id, bboxes in tracks.items():
                # 每个ID可能有多个bbox, 取第i个(如果有)
                if i < len(bboxes):
                    x1, y1, x2, y2 = bboxes[i]
                    px1, py1, px2, py2 = qwen_bbox_to_pixel(x1, y1, x2, y2, img_width, img_height)
                    pred_bboxes_pixel.append((px1, py1, px2, py2))
                    pred_ids.append(track_id)

            # NMS: 去除重叠框
            pred_bboxes_nms = nms(pred_bboxes_pixel, iou_threshold=0.5)
            # 保留NMS后的ID
            nms_ids = []
            for bbox in pred_bboxes_nms:
                idx = pred_bboxes_pixel.index(bbox)
                nms_ids.append(pred_ids[idx])
            pred_bboxes_pixel = pred_bboxes_nms
            pred_ids = nms_ids

            # 计算 IoU: 每个 pred bbox 与最佳 GT bbox 匹配
            frame_ious = []
            for pred_bbox in pred_bboxes_pixel:
                best_iou = 0.0
                for gt_bbox in gt_bboxes_pixel:
                    iou = compute_iou(pred_bbox, gt_bbox)
                    best_iou = max(best_iou, iou)
                frame_ious.append(best_iou)
                total_iou += best_iou
                total_iou_count += 1

            # 计算 ID 匹配率: 基于 IoU>0.3 的匹配, 检查 ID 是否一致
            for pi, pred_bbox in enumerate(pred_bboxes_pixel):
                for gi, gt_bbox in enumerate(gt_bboxes_pixel):
                    if compute_iou(pred_bbox, gt_bbox) > 0.3:
                        # 空间匹配成功, 比较 ID
                        total_id_count += 1
                        if pred_ids[pi] == gt_ids[gi]:
                            total_id_match += 1
                        break

            all_frame_data.append({
                'frame_id': frame_id,
                'image_path': image_path,
                'gt_bboxes': gt_bboxes_pixel,
                'gt_ids': gt_ids,
                'pred_bboxes': pred_bboxes_pixel,
                'pred_ids': pred_ids,
                'ious': frame_ious,
            })

            avg_iou = np.mean(frame_ious) if frame_ious else 0.0
            print(f"    帧{frame_id}: GT={len(gt_bboxes_pixel)}, Pred={len(pred_bboxes_pixel)}, "
                  f"平均IoU={avg_iou:.3f}")

        # ---- 绘制网格可视化 (2行5列) ----
        cols = 5
        rows = 2
        thumb_w = img_width // 2
        thumb_h = img_height // 2
        grid_w = cols * thumb_w + (cols + 1) * 5
        grid_h = rows * thumb_h + (rows + 1) * 5 + 30
        grid = Image.new('RGB', (grid_w, grid_h), 'white')
        grid_draw = ImageDraw.Draw(grid)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        import colorsys

        # 为 pred track ID 生成颜色
        id_colors = {}
        for track_id in tracks:
            hue = (track_id * 0.618) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            id_colors[track_id] = (int(r * 255), int(g * 255), int(b * 255))

        for idx, fd in enumerate(all_frame_data):
            row = idx // cols
            col = idx % cols
            x_off = 5 + col * (thumb_w + 5)
            y_off = 5 + row * (thumb_h + 5 + 20)

            image = Image.open(fd['image_path']).convert("RGB")
            image = image.resize((thumb_w, thumb_h), Image.LANCZOS)
            draw = ImageDraw.Draw(image)

            # 缩放因子
            sx = thumb_w / img_width
            sy = thumb_h / img_height

            # 绘制 GT (绿色)
            for bbox in fd['gt_bboxes']:
                px1, py1, px2, py2 = int(bbox[0]*sx), int(bbox[1]*sy), int(bbox[2]*sx), int(bbox[3]*sy)
                draw.rectangle([px1, py1, px2, py2], outline='green', width=2)

            # 绘制 Pred (按 ID 着色)
            for pi, bbox in enumerate(fd['pred_bboxes']):
                tid = fd['pred_ids'][pi]
                color = id_colors.get(tid, 'red')
                px1, py1, px2, py2 = int(bbox[0]*sx), int(bbox[1]*sy), int(bbox[2]*sx), int(bbox[3]*sy)
                draw.rectangle([px1, py1, px2, py2], outline=color, width=2)
                draw.text((px1, py1 - 12), f"ID{tid}", fill=color, font=font)

            grid.paste(image, (x_off, y_off))

            # 帧标题
            avg_iou = np.mean(fd['ious']) if fd['ious'] else 0.0
            grid_draw.text((x_off, y_off + thumb_h + 2),
                          f"F{fd['frame_id']} IoU={avg_iou:.2f}", fill='black', font=font)

        # 添加图例
        grid_draw.text((5, grid_h - 25), "Green=GT  Colored=Pred(by ID)", fill='black', font=font)

        grid_save = str(output_dir / f"{seq_name}_grid.jpg")
        grid.save(grid_save, quality=90)
        print(f"\n    网格图保存: {grid_save}")

        # ---- 绘制逐帧对比图 ----
        per_frame_dir = output_dir / seq_name
        per_frame_dir.mkdir(parents=True, exist_ok=True)

        for fd in all_frame_data:
            image = Image.open(fd['image_path']).convert("RGB")

            # GT 图 (绿色)
            gt_image = image.copy()
            gt_draw = ImageDraw.Draw(gt_image)
            for gi, bbox in enumerate(fd['gt_bboxes']):
                gt_draw.rectangle(list(bbox), outline='green', width=2)
                gid = fd['gt_ids'][gi]
                gt_draw.text((bbox[0], bbox[1] - 18), f"GT-ID{gid}", fill='green', font=font)

            # Pred 图 (按 ID 着色)
            pred_image = image.copy()
            pred_draw = ImageDraw.Draw(pred_image)
            for pi, bbox in enumerate(fd['pred_bboxes']):
                tid = fd['pred_ids'][pi]
                color = id_colors.get(tid, 'red')
                pred_draw.rectangle(list(bbox), outline=color, width=2)
                pred_draw.text((bbox[0], bbox[1] - 18), f"Pred-ID{tid}", fill=color, font=font)

            # 合并
            combined = Image.new('RGB', (img_width * 2 + 20, img_height + 40), 'white')
            combined.paste(gt_image, (0, 0))
            combined.paste(pred_image, (img_width + 20, 0))

            c_draw = ImageDraw.Draw(combined)
            try:
                c_font = ImageFont.truetype("arial.ttf", 20)
            except:
                c_font = ImageFont.load_default()
            avg_iou = np.mean(fd['ious']) if fd['ious'] else 0.0
            c_draw.text((10, img_height + 5),
                       f"GT (Green) - {len(fd['gt_bboxes'])} persons", fill='green', font=c_font)
            c_draw.text((img_width + 30, img_height + 5),
                       f"Pred (Colored) - {len(fd['pred_bboxes'])} persons, avgIoU={avg_iou:.3f}",
                       fill='red', font=c_font)

            save_name = f"frame{fd['frame_id']:06d}.jpg"
            combined.save(str(per_frame_dir / save_name), quality=90)

        print(f"    逐帧对比图保存: {per_frame_dir}")

    # ---- 汇总指标 ----
    print(f"\n{'=' * 60}")
    print(f"10帧跟踪评估汇总")
    print(f"{'=' * 60}")
    if total_iou_count > 0:
        print(f"  平均 IoU (Pred→GT最佳匹配): {total_iou / total_iou_count:.4f}")
    if total_id_count > 0:
        print(f"  ID 匹配率 (IoU>0.3时ID一致): {total_id_match / total_id_count * 100:.1f}% "
              f"({total_id_match}/{total_id_count})")
    print(f"  总 Pred bbox 数: {total_iou_count}")


def visualize_compare(model, processor, lora_path, output_dir):
    """对比微调前后的检测效果"""
    print("\n" + "=" * 60)
    print("微调前后对比可视化")
    print("=" * 60)

    output_dir = Path(output_dir) / "compare"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 选择一个测试帧
    seq_names = sorted([d for d in os.listdir(MOT17_PATH)
                        if (MOT17_PATH / d).is_dir()
                        and (MOT17_PATH / d / "gt" / "gt.txt").exists()])

    # 序列过滤
    if EVAL_SEQ_FILTER:
        seq_names = [s for s in seq_names if s in EVAL_SEQ_FILTER]

    if not seq_names:
        print("  ❌ 没有可用的 MOT17 序列")
        return

    seq_path = MOT17_PATH / seq_names[0]
    img_dir = seq_path / "img1"
    frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    image_path = str(img_dir / frames[0])

    # 原始模型推理
    print("  原始模型推理...")
    response_before = run_detection(model, processor, image_path)
    bboxes_before = parse_bboxes_from_text(response_before)
    print(f"    检测到 {len(bboxes_before)} 个bbox")

    # 释放原始模型显存
    print("  释放原始模型显存...")
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 加载 LoRA 模型
    print("  加载 LoRA 模型...")
    model_lora, processor_lora = load_model(MODEL_PATH, lora_path=lora_path)
    print("  LoRA 模型推理...")
    response_after = run_detection(model_lora, processor_lora, image_path)
    bboxes_after = parse_bboxes_from_text(response_after)
    print(f"    检测到 {len(bboxes_after)} 个bbox")

    # 绘制对比
    image = Image.open(image_path).convert("RGB")
    info = {}
    seqinfo_path = seq_path / "seqinfo.ini"
    if seqinfo_path.exists():
        with open(seqinfo_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, val = line.split('=', 1)
                    info[key.strip()] = val.strip()
    img_width = int(info.get('imWidth', 1920))
    img_height = int(info.get('imHeight', 1080))

    before_image = image.copy()
    draw_bboxes_on_image(before_image, bboxes_before, color='red', label='Before',
                        img_width=img_width, img_height=img_height)

    after_image = image.copy()
    draw_bboxes_on_image(after_image, bboxes_after, color='blue', label='After',
                        img_width=img_width, img_height=img_height)

    combined = Image.new('RGB', (img_width * 2 + 20, img_height + 40), 'white')
    combined.paste(before_image, (0, 0))
    combined.paste(after_image, (img_width + 20, 0))

    draw = ImageDraw.Draw(combined)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((10, img_height + 5), f"Before LoRA ({len(bboxes_before)} det)", fill='red', font=font)
    draw.text((img_width + 30, img_height + 5), f"After LoRA ({len(bboxes_after)} det)", fill='blue', font=font)

    combined.save(str(output_dir / "compare_before_after.jpg"))
    print(f"\n  ✅ 保存: {output_dir / 'compare_before_after.jpg'}")
    print(f"  微调前检测数: {len(bboxes_before)}")
    print(f"  微调后检测数: {len(bboxes_after)}")

    del model_lora
    torch.cuda.empty_cache()


def visualize_detect_match(model, processor, output_dir):
    """先检测再匹配 (Detect-then-Match) 可视化"""

    from model_method.build_sft_data import parse_gt, get_sequence_info

    print("\n" + "=" * 60)
    print("先检测再匹配 (Detect-then-Match) 可视化")
    print("=" * 60)

    output_dir = Path(output_dir) / "track_dm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历MOT17序列
    seq_dir = MOT17_PATH
    if not seq_dir.exists():
        print(f"❌ MOT17 路径不存在: {seq_dir}")
        return

    total_tp, total_fp, total_fn = 0, 0, 0
    total_gt_count, total_pred_count = 0, 0
    # MOT 格式预测结果: {frame_id: {track_id: [x1,y1,x2,y2]}}
    mot_predictions = {}
    mot_confidences = {}

    for seq_name in sorted(os.listdir(seq_dir)):
        seq_path = seq_dir / seq_name
        if not seq_path.is_dir():
            continue

        # 序列过滤
        if EVAL_SEQ_FILTER and seq_name not in EVAL_SEQ_FILTER:
            continue

        img_dir = seq_path / "img1"
        gt_path = seq_path / "gt" / "gt.txt"

        if not gt_path.exists() or not img_dir.exists():
            continue

        # 读取序列信息
        info = get_sequence_info(seq_path)
        img_width = int(info.get('imWidth', 1920))
        img_height = int(info.get('imHeight', 1080))

        # 解析GT
        annotations = parse_gt(gt_path)

        # 选择连续帧
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        _default_n = 10
        _max_n = EVAL_MAX_FRAMES if EVAL_MAX_FRAMES else _default_n
        num_frames = min(_max_n, len(frames))
        image_paths = [str(img_dir / frames[i]) for i in range(num_frames)]

        if len(image_paths) < 2:
            continue

        print(f"\n  {seq_name}: {num_frames} 连续帧 (帧1-{num_frames})")

        # 运行detect-match推理
        frame_detections, frame_ids = run_detect_match_tracking(model, processor, image_paths)

        # 为每帧构建GT和Pred的bbox列表
        all_frame_data = []
        total_iou = 0.0
        total_iou_count = 0

        for i, image_path in enumerate(tqdm(image_paths, desc="    解析", unit="帧")):
            frame_file = frames[i]
            frame_id = int(frame_file.replace('.jpg', ''))

            # GT bboxes (像素坐标)
            gt_anns = annotations.get(frame_id, [])
            gt_bboxes_pixel = []
            gt_ids = []
            for ann in gt_anns:
                x, y, w, h = ann['bbox']
                gt_bboxes_pixel.append((int(x), int(y), int(x + w), int(y + h)))
                gt_ids.append(ann.get('id', -1))

            # Pred bboxes (从Qwen 0-1000坐标转像素坐标)
            pred_bboxes_pixel = []
            pred_ids = []
            if frame_detections[i] and frame_ids[i]:
                for k, bbox in enumerate(frame_detections[i]):
                    x1, y1, x2, y2 = bbox
                    px1, py1, px2, py2 = qwen_bbox_to_pixel(x1, y1, x2, y2, img_width, img_height)
                    pred_bboxes_pixel.append((px1, py1, px2, py2))
                    pred_ids.append(frame_ids[i][k])

            # NMS
            pred_bboxes_nms = nms(pred_bboxes_pixel, iou_threshold=0.5)
            nms_ids = []
            for bbox in pred_bboxes_nms:
                idx = pred_bboxes_pixel.index(bbox)
                nms_ids.append(pred_ids[idx])
            pred_bboxes_pixel = pred_bboxes_nms
            pred_ids = nms_ids

            # 收集 MOT 格式预测 (1-based frame_id)
            mot_predictions[frame_id] = {}
            mot_confidences[frame_id] = {}
            for k, bbox in enumerate(pred_bboxes_pixel):
                tid = pred_ids[k]
                x1, y1, x2, y2 = bbox
                mot_predictions[frame_id][tid] = np.array([x1, y1, x2, y2], dtype=np.float32)
                mot_confidences[frame_id][tid] = 1.0

            # 计算IoU
            frame_ious = []
            for pred_bbox in pred_bboxes_pixel:
                best_iou = 0.0
                for gt_bbox in gt_bboxes_pixel:
                    iou = compute_iou(pred_bbox, gt_bbox)
                    best_iou = max(best_iou, iou)
                frame_ious.append(best_iou)
                total_iou += best_iou
                total_iou_count += 1

            # TP/FP/FN
            matched_gt = set()
            tp = 0
            for pred_bbox in pred_bboxes_pixel:
                for gi, gt_bbox in enumerate(gt_bboxes_pixel):
                    if gi in matched_gt:
                        continue
                    if compute_iou(pred_bbox, gt_bbox) > 0.5:
                        tp += 1
                        matched_gt.add(gi)
                        break
            fp = len(pred_bboxes_pixel) - tp
            fn = len(gt_bboxes_pixel) - tp

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt_count += len(gt_bboxes_pixel)
            total_pred_count += len(pred_bboxes_pixel)

            all_frame_data.append({
                'frame_id': frame_id,
                'image_path': image_path,
                'gt_bboxes': gt_bboxes_pixel,
                'gt_ids': gt_ids,
                'pred_bboxes': pred_bboxes_pixel,
                'pred_ids': pred_ids,
                'ious': frame_ious,
            })

            avg_iou = np.mean(frame_ious) if frame_ious else 0.0
            print(f"    帧{frame_id}: GT={len(gt_bboxes_pixel)}, Pred={len(pred_bboxes_pixel)}, "
                  f"TP={tp}, FP={fp}, FN={fn}, 平均IoU={avg_iou:.3f}")

        # 绘制逐帧对比图 (GT 和 Pred 左右拼接)
        import colorsys
        seq_out_dir = output_dir / seq_name
        seq_out_dir.mkdir(parents=True, exist_ok=True)

        # 为pred track ID生成颜色
        all_pred_ids = set()
        for fd in all_frame_data:
            all_pred_ids.update(fd['pred_ids'])

        id_colors = {}
        for tid in all_pred_ids:
            hue = (tid * 0.618) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            id_colors[tid] = (int(r * 255), int(g * 255), int(b * 255))

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        for fd in all_frame_data:
            image = Image.open(fd['image_path']).convert("RGB")

            # GT 图 (绿色) - 左侧
            gt_image = image.copy()
            gt_draw = ImageDraw.Draw(gt_image)
            for gi, bbox in enumerate(fd['gt_bboxes']):
                gt_draw.rectangle(list(bbox), outline='green', width=2)
                gid = fd['gt_ids'][gi]
                gt_draw.text((bbox[0], bbox[1] - 14), f"GT{gid}", fill='green', font=font)

            # Pred 图 (按ID着色) - 右侧
            pred_image = image.copy()
            pred_draw = ImageDraw.Draw(pred_image)
            for pi, bbox in enumerate(fd['pred_bboxes']):
                tid = fd['pred_ids'][pi]
                color = id_colors.get(tid, 'red')
                pred_draw.rectangle(list(bbox), outline=color, width=2)
                pred_draw.text((bbox[0], bbox[1] - 14), f"ID{tid}", fill=color, font=font)

            # 左右拼接
            combined = Image.new('RGB', (img_width * 2 + 20, img_height + 40), 'white')
            combined.paste(gt_image, (0, 0))
            combined.paste(pred_image, (img_width + 20, 0))

            c_draw = ImageDraw.Draw(combined)
            try:
                c_font = ImageFont.truetype("arial.ttf", 20)
            except:
                c_font = ImageFont.load_default()
            avg_iou = np.mean(fd['ious']) if fd['ious'] else 0.0
            c_draw.text((10, img_height + 5),
                       f"GT (Green) - {len(fd['gt_bboxes'])} persons",
                       fill='green', font=c_font)
            c_draw.text((img_width + 30, img_height + 5),
                       f"Pred (Colored) - {len(fd['pred_bboxes'])} persons, avgIoU={avg_iou:.3f}",
                       fill='red', font=c_font)

            save_path = seq_out_dir / f"frame{fd['frame_id']:06d}.jpg"
            combined.save(str(save_path), quality=90)

        print(f"    逐帧对比图保存: {seq_out_dir}")

    # 使用 TrackingEvaluator 计算标准 MOT 指标
    from tools.evaluate import TrackingEvaluator
    evaluator = TrackingEvaluator(iou_threshold=0.5)
    # 加载 GT (所有序列)
    gt_all = {}
    for seq_name in sorted(os.listdir(seq_dir)):
        seq_path = seq_dir / seq_name
        if not seq_path.is_dir():
            continue
        if EVAL_SEQ_FILTER and seq_name not in EVAL_SEQ_FILTER:
            continue
        gt_path = seq_path / "gt" / "gt.txt"
        if gt_path.exists():
            gt = evaluator.load_mot_gt(gt_path, class_ids=[1])
            gt_all.update(gt)

    # 计算标准 MOT 指标
    metrics_obj = evaluator.evaluate(mot_predictions, gt_all, eval_mode="prediction_range")
    metrics = metrics_obj.to_dict()

    # 保存 MOT 格式预测结果
    evaluator.save_predictions_mot(mot_predictions, str(output_dir / "predictions.txt"), mot_confidences)

    # 保存评估报告
    with open(str(output_dir / "evaluation_report.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 汇总
    print("\n" + "=" * 60)
    print("Detect-Match 跟踪评估汇总")
    print("=" * 60)
    print(f"  总 GT 行人数: {total_gt_count}")
    print(f"  总预测行人数: {total_pred_count}")
    print(f"  True Positives (TP): {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\n  精确率 (Precision): {precision*100:.1f}%")
    print(f"  召回率 (Recall): {recall*100:.1f}%")
    print(f"  F1 分数: {f1*100:.1f}%")
    print(f"\n  标准 MOT 指标:")
    print(f"  MOTA: {metrics['MOTA']*100:.1f}%")
    print(f"  MOTP: {metrics['MOTP']*100:.1f}%")
    print(f"  IDF1: {metrics['IDF1']*100:.1f}%")
    print(f"  IDSW: {metrics['IDSW']}")

    print(f"\n可视化结果保存在: {output_dir}")


def visualize_detect_track(model, processor, output_dir):
    """
    检测 + IoU 后处理跟踪可视化 (方案 B Stage 2)

    模型只做单帧检测 (原生格式), 跨帧跟踪由 IoU 后处理完成。
    评估指标: 检测 TP/FP/FN + 跟踪 ID 一致性 (IDSW)。
    """

    from model_method.build_sft_data import parse_gt, get_sequence_info

    print("\n" + "=" * 60)
    print("检测 + IoU 后处理跟踪 (方案 B Stage 2)")
    print("=" * 60)

    output_dir = Path(output_dir) / "track_detect_iou"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 遍历 MOT17 序列
    seq_dir = MOT17_PATH
    if not seq_dir.exists():
        print(f"❌ MOT17 路径不存在: {seq_dir}")
        return

    total_tp, total_fp, total_fn = 0, 0, 0
    total_gt_count, total_pred_count = 0, 0
    total_idsw = 0  # ID 切换次数
    total_id_match_count = 0  # ID 匹配总数 (用于计算 ID 保持率)
    # 每序列预测结果和指标 (避免跨序列 frame_id 冲突)
    per_sequence_data = {}  # {seq_name: {'predictions': {...}, 'confidences': {...}}}
    per_sequence_metrics = {}  # {seq_name: {metrics dict}}

    for seq_name in sorted(os.listdir(seq_dir)):
        seq_path = seq_dir / seq_name
        if not seq_path.is_dir():
            continue

        # 序列过滤
        if EVAL_SEQ_FILTER and seq_name not in EVAL_SEQ_FILTER:
            continue

        img_dir = seq_path / "img1"
        gt_path = seq_path / "gt" / "gt.txt"

        if not gt_path.exists() or not img_dir.exists():
            continue

        # 读取序列信息
        info = get_sequence_info(seq_path)
        img_width = int(info.get('imWidth', 1920))
        img_height = int(info.get('imHeight', 1080))

        # 解析 GT
        annotations = parse_gt(gt_path)

        # 选择连续帧
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        _default_n = 10
        _max_n = EVAL_MAX_FRAMES if EVAL_MAX_FRAMES else _default_n
        num_frames = min(_max_n, len(frames))
        image_paths = [str(img_dir / frames[i]) for i in range(num_frames)]

        if len(image_paths) < 2:
            continue

        print(f"\n  {seq_name}: {num_frames} 连续帧 (帧1-{num_frames})")

        # 运行检测 + IoU 跟踪
        frame_detections, frame_ids = detect_and_track_iou(model, processor, image_paths)

        # 为每帧构建 GT 和 Pred 的 bbox 列表
        all_frame_data = []
        # 当前序列的 MOT 格式预测
        seq_predictions = {}
        seq_confidences = {}

        # 用于计算 IDSW: 记录每个 GT track_id 在上一帧匹配的 pred_id
        prev_gt_to_pred = {}  # {gt_id: pred_id}

        for i, image_path in enumerate(tqdm(image_paths, desc="    解析", unit="帧")):
            frame_file = frames[i]
            frame_id = int(frame_file.replace('.jpg', ''))

            # GT bboxes (像素坐标)
            gt_anns = annotations.get(frame_id, [])
            gt_bboxes_pixel = []
            gt_ids = []
            for ann in gt_anns:
                x, y, w, h = ann['bbox']
                gt_bboxes_pixel.append((int(x), int(y), int(x + w), int(y + h)))
                gt_ids.append(ann.get('id', -1))

            # Pred bboxes (从 Qwen 0-1000 坐标转像素坐标)
            pred_bboxes_pixel = []
            pred_ids = []
            if frame_detections[i] and frame_ids[i]:
                for k, bbox in enumerate(frame_detections[i]):
                    x1, y1, x2, y2 = bbox
                    px1, py1, px2, py2 = qwen_bbox_to_pixel(x1, y1, x2, y2, img_width, img_height)
                    pred_bboxes_pixel.append((px1, py1, px2, py2))
                    pred_ids.append(frame_ids[i][k])

            # 收集当前序列的 MOT 格式预测
            seq_predictions[frame_id] = {}
            seq_confidences[frame_id] = {}
            for k, bbox in enumerate(pred_bboxes_pixel):
                tid = pred_ids[k]
                x1, y1, x2, y2 = bbox
                seq_predictions[frame_id][tid] = np.array([x1, y1, x2, y2], dtype=np.float32)
                seq_confidences[frame_id][tid] = 1.0

            # TP/FP/FN (IoU 阈值=0.5)
            matched_gt = set()
            matched_pred = set()
            tp = 0
            # 贪心匹配: 按 IoU 从高到低
            iou_pairs = []
            for pi, pb in enumerate(pred_bboxes_pixel):
                for gi, gb in enumerate(gt_bboxes_pixel):
                    iou = compute_iou(pb, gb)
                    if iou > 0.5:
                        iou_pairs.append((iou, pi, gi))
            iou_pairs.sort(reverse=True)
            for iou, pi, gi in iou_pairs:
                if pi in matched_pred or gi in matched_gt:
                    continue
                matched_pred.add(pi)
                matched_gt.add(gi)
                tp += 1

                # IDSW 检测: 如果该 GT id 上一帧匹配了不同的 pred_id, 则 IDSW+1
                gt_id = gt_ids[gi]
                pred_id = pred_ids[pi]
                if gt_id in prev_gt_to_pred:
                    if prev_gt_to_pred[gt_id] != pred_id:
                        total_idsw += 1
                    total_id_match_count += 1
                prev_gt_to_pred[gt_id] = pred_id

            fp = len(pred_bboxes_pixel) - tp
            fn = len(gt_bboxes_pixel) - tp

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_gt_count += len(gt_bboxes_pixel)
            total_pred_count += len(pred_bboxes_pixel)

            all_frame_data.append({
                'frame_id': frame_id,
                'image_path': image_path,
                'gt_bboxes': gt_bboxes_pixel,
                'gt_ids': gt_ids,
                'pred_bboxes': pred_bboxes_pixel,
                'pred_ids': pred_ids,
                'tp': tp, 'fp': fp, 'fn': fn,
            })

            print(f"    帧{frame_id}: GT={len(gt_bboxes_pixel)}, Pred={len(pred_bboxes_pixel)}, "
                  f"TP={tp}, FP={fp}, FN={fn}")

        # 计算当前序列的标准 MOT 指标
        from tools.evaluate import TrackingEvaluator
        evaluator = TrackingEvaluator(iou_threshold=0.5)
        seq_gt = evaluator.load_mot_gt(gt_path, class_ids=[1])
        seq_metrics_obj = evaluator.evaluate(seq_predictions, seq_gt, eval_mode="prediction_range")
        seq_metrics = seq_metrics_obj.to_dict()
        per_sequence_metrics[seq_name] = seq_metrics
        per_sequence_data[seq_name] = {
            'predictions': seq_predictions,
            'confidences': seq_confidences,
        }

        # 保存当前序列的 MOT 格式预测
        seq_out_dir = output_dir / seq_name
        seq_out_dir.mkdir(parents=True, exist_ok=True)
        evaluator.save_predictions_mot(
            seq_predictions,
            str(seq_out_dir / "predictions.txt"),
            seq_confidences,
        )

        # 绘制逐帧对比图 (GT 和 Pred 左右拼接)
        import colorsys

        # 为 pred track ID 生成颜色
        all_pred_ids = set()
        for fd in all_frame_data:
            all_pred_ids.update(fd['pred_ids'])

        id_colors = {}
        for tid in all_pred_ids:
            hue = (tid * 0.618) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            id_colors[tid] = (int(r * 255), int(g * 255), int(b * 255))

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        for fd in all_frame_data:
            image = Image.open(fd['image_path']).convert("RGB")

            # GT 图 (绿色) - 左侧
            gt_image = image.copy()
            gt_draw = ImageDraw.Draw(gt_image)
            for gi, bbox in enumerate(fd['gt_bboxes']):
                gt_draw.rectangle(list(bbox), outline='green', width=2)
                gid = fd['gt_ids'][gi]
                gt_draw.text((bbox[0], bbox[1] - 14), f"GT{gid}", fill='green', font=font)

            # Pred 图 (按 ID 着色) - 右侧
            pred_image = image.copy()
            pred_draw = ImageDraw.Draw(pred_image)
            for pi, bbox in enumerate(fd['pred_bboxes']):
                tid = fd['pred_ids'][pi]
                color = id_colors.get(tid, 'red')
                pred_draw.rectangle(list(bbox), outline=color, width=2)
                pred_draw.text((bbox[0], bbox[1] - 14), f"ID{tid}", fill=color, font=font)

            # 左右拼接
            combined = Image.new('RGB', (img_width * 2 + 20, img_height + 40), 'white')
            combined.paste(gt_image, (0, 0))
            combined.paste(pred_image, (img_width + 20, 0))

            c_draw = ImageDraw.Draw(combined)
            try:
                c_font = ImageFont.truetype("arial.ttf", 20)
            except:
                c_font = ImageFont.load_default()
            c_draw.text((10, img_height + 5),
                       f"GT (Green) - {len(fd['gt_bboxes'])} persons",
                       fill='green', font=c_font)
            c_draw.text((img_width + 30, img_height + 5),
                       f"Pred (Colored) - {len(fd['pred_bboxes'])} persons",
                       fill='red', font=c_font)

            save_path = seq_out_dir / f"frame{fd['frame_id']:06d}.jpg"
            combined.save(str(save_path), quality=90)

        print(f"    逐帧对比图保存: {seq_out_dir}")

    # 计算汇总指标 (从 per-sequence 累加)
    total_metrics = {
        'MOTA': 0.0, 'MOTP': 0.0, 'IDF1': 0.0,
        'IDSW': 0, 'FP': 0, 'FN': 0, 'TP': 0, 'GT': 0,
        'MT': 0, 'ML': 0, 'Frag': 0,
        'num_frames': 0, 'num_gt_ids': 0, 'num_pred_ids': 0,
    }
    motp_weighted_sum = 0.0  # MOTP 加权平均 (按 TP 加权)
    motp_total_tp = 0
    for seq_name, seq_m in per_sequence_metrics.items():
        for k in total_metrics:
            if isinstance(total_metrics[k], (int, float)):
                total_metrics[k] += seq_m.get(k, 0)
        # MOTP 按 TP 加权
        motp_weighted_sum += seq_m.get('MOTP', 0) * seq_m.get('TP', 0)
        motp_total_tp += seq_m.get('TP', 0)
    # 重新计算 MOTA (基于累加的 FP/FN/IDSW/GT)
    if total_metrics['GT'] > 0:
        total_metrics['MOTA'] = 1 - (total_metrics['FP'] + total_metrics['FN'] + total_metrics['IDSW']) / total_metrics['GT']
    # MOTP 加权平均
    if motp_total_tp > 0:
        total_metrics['MOTP'] = motp_weighted_sum / motp_total_tp
    # IDF1 简单平均
    if per_sequence_metrics:
        total_metrics['IDF1'] = sum(m.get('IDF1', 0) for m in per_sequence_metrics.values()) / len(per_sequence_metrics)
    # Precision/Recall
    total_metrics['Precision'] = total_metrics['TP'] / max(total_metrics['TP'] + total_metrics['FP'], 1)
    total_metrics['Recall'] = total_metrics['TP'] / max(total_metrics['TP'] + total_metrics['FN'], 1)

    # 保存评估报告 (包含 per-sequence 和汇总)
    report = {
        'overall': total_metrics,
        'per_sequence': per_sequence_metrics,
    }
    with open(str(output_dir / "evaluation_report.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 汇总
    print("\n" + "=" * 60)
    print("检测 + IoU 跟踪 评估汇总")
    print("=" * 60)
    print(f"  总 GT 行人数: {total_gt_count}")
    print(f"  总预测行人数: {total_pred_count}")
    print(f"  True Positives (TP): {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"\n  精确率 (Precision): {precision*100:.1f}%")
    print(f"  召回率 (Recall): {recall*100:.1f}%")
    print(f"  F1 分数: {f1*100:.1f}%")
    print(f"\n  ID 切换次数 (IDSW): {total_idsw}")
    if total_id_match_count > 0:
        id_keep_rate = (total_id_match_count - total_idsw) / total_id_match_count * 100
        print(f"  ID 保持率: {id_keep_rate:.1f}% ({total_id_match_count - total_idsw}/{total_id_match_count})")
    print(f"\n  标准 MOT 指标 (汇总):")
    print(f"  MOTA: {total_metrics['MOTA']*100:.1f}%")
    print(f"  MOTP: {total_metrics['MOTP']*100:.1f}%")
    print(f"  IDF1: {total_metrics['IDF1']*100:.1f}%")
    print(f"  IDSW (标准): {total_metrics['IDSW']}")

    # 打印 per-sequence 指标
    print(f"\n  Per-sequence 指标:")
    for seq_name, seq_m in per_sequence_metrics.items():
        print(f"    {seq_name}: MOTA={seq_m['MOTA']*100:.1f}%, MOTP={seq_m['MOTP']*100:.1f}%, "
              f"IDF1={seq_m['IDF1']*100:.1f}%, IDSW={seq_m['IDSW']}, "
              f"TP={seq_m['TP']}, FP={seq_m['FP']}, FN={seq_m['FN']}")

    print(f"\n可视化结果保存在: {output_dir}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen2-VL 训练前/后可视化")
    parser.add_argument(
        '--mode', type=str, default='detect',
        choices=['detect', 'track', 'track10', 'track_dm', 'detect_track', 'compare'],
        help='可视化模式: detect=单帧检测, track=多帧跟踪, track10=10帧连续跟踪, track_dm=先检测再匹配, detect_track=检测+IoU后处理跟踪(方案B), compare=微调前后对比'
    )
    parser.add_argument(
        '--model-path', type=str, default=str(MODEL_PATH),
        help='模型路径'
    )
    parser.add_argument(
        '--lora-path', type=str, default=None,
        help='LoRA 权重路径 (compare 模式必需)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=str(OUTPUT_DIR),
        help='输出目录'
    )
    parser.add_argument(
        '--seq-filter', type=str, default='MOT17-02-FRCNN,MOT17-11-FRCNN',
        help='只测试指定序列，逗号分隔 (默认: MOT17-02-FRCNN,MOT17-11-FRCNN)'
    )
    parser.add_argument(
        '--max-frames', type=int, default=10,
        help='每个序列最多测试的帧数 (默认: 10)'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen2-VL 行人跟踪 - 可视化")
    print("=" * 60)
    print(f"  模式: {args.mode}")
    print(f"  模型: {args.model_path}")
    if args.lora_path:
        print(f"  LoRA: {args.lora_path}")
    print(f"  输出: {args.output_dir}")
    if args.seq_filter:
        print(f"  序列过滤: {args.seq_filter}")
    if args.max_frames:
        print(f"  最大帧数: {args.max_frames}")
    print()

    # 全局评估参数 (供各可视化函数读取)
    global EVAL_SEQ_FILTER, EVAL_MAX_FRAMES
    EVAL_SEQ_FILTER = [s.strip() for s in args.seq_filter.split(',')] if args.seq_filter else None
    EVAL_MAX_FRAMES = args.max_frames

    # 加载模型 (所有模式都支持 --lora-path)
    model, processor = load_model(args.model_path, lora_path=args.lora_path)

    if args.mode == 'detect':
        visualize_detection(model, processor, args.output_dir)
    elif args.mode == 'track':
        visualize_tracking(model, processor, args.output_dir)
    elif args.mode == 'track10':
        visualize_track10(model, processor, args.output_dir)
    elif args.mode == 'track_dm':
        visualize_detect_match(model, processor, args.output_dir)
    elif args.mode == 'detect_track':
        visualize_detect_track(model, processor, args.output_dir)
    elif args.mode == 'compare':
        if not args.lora_path:
            print("❌ compare 模式需要 --lora-path 参数")
            sys.exit(1)
        visualize_compare(model, processor, args.lora_path, args.output_dir)

    print(f"\n可视化结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()
