#!/usr/bin/env python3
"""
重新训练 HOG+SVM 行人检测器
使用自定义场景标注数据 (scene1-4)，替换 INRIA 预训练权重

流程:
  1. 从标注帧提取正样本(行人区域)和负样本(背景区域)
  2. 提取 HOG 特征 (3780维)
  3. 训练 LinearSVC
  4. 可选: Hard Negative Mining
  5. 保存为 OpenCV 兼容格式
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import argparse
import time
import sys
import re

WIN_SIZE = (64, 128)
CELL_SIZE = (8, 8)
BLOCK_SIZE = (16, 16)
BLOCK_STRIDE = (8, 8)


def compute_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / max(area1 + area2 - inter, 1)


def extract_samples(video_path, labels_dir, pos_per_frame=8, neg_per_frame=15, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return [], []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {video_path.name} ({w_vid}x{h_vid}, {total} frames)")

    label_files = sorted(Path(labels_dir).glob("*.txt"))
    if max_frames:
        label_files = label_files[:max_frames]
    print(f"  Labels: {len(label_files)} frames")

    frame_map = {}
    for lf in label_files:
        nums = re.findall(r'\d+', lf.stem)
        if nums:
            frame_map[int(nums[-1])] = lf

    positives, negatives = [], []
    frame_count = 0

    for frame_id in sorted(frame_map.keys()):
        lf = frame_map[frame_id]

        # Read GT boxes
        gt_boxes = []
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id != 0:
                    continue
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw/2) * w_vid)
                y1 = int((cy - bh/2) * h_vid)
                x2 = int((cx + bw/2) * w_vid)
                y2 = int((cy + bh/2) * h_vid)
                bw_px, bh_px = x2 - x1, y2 - y1
                if bw_px >= 16 and bh_px >= 32:
                    gt_boxes.append((x1, y1, x2, y2))

        if not gt_boxes:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        frame_count += 1

        # Positive samples: crop pedestrian regions
        for (x1, y1, x2, y2) in gt_boxes[:pos_per_frame]:
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            crop = gray[y1c:y2c, x1c:x2c]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 16:
                continue
            crop = cv2.resize(crop, WIN_SIZE)
            positives.append(crop)

        # Negative samples: random background regions
        neg_count = 0
        for _ in range(neg_per_frame * 5):
            if neg_count >= neg_per_frame:
                break
            nx = np.random.randint(0, max(1, w - 64))
            ny = np.random.randint(0, max(1, h - 128))
            neg_box = [nx, ny, nx+64, ny+128]
            overlap = any(compute_iou(neg_box, gb) > 0.1 for gb in gt_boxes)
            if not overlap:
                crop = gray[ny:ny+128, nx:nx+64]
                negatives.append(crop)
                neg_count += 1

        if frame_count % 50 == 0:
            print(f"  Processed {frame_count} frames, pos={len(positives)}, neg={len(negatives)}")

    cap.release()
    return positives, negatives


def extract_hog_features(samples):
    hog = cv2.HOGDescriptor(
        _winSize=WIN_SIZE, _blockSize=BLOCK_SIZE,
        _blockStride=BLOCK_STRIDE, _cellSize=CELL_SIZE, _nbins=9
    )
    features = []
    for i, sample in enumerate(samples):
        feat = hog.compute(sample)
        features.append(feat.flatten())
        if (i + 1) % 2000 == 0:
            print(f"  HOG: {i+1}/{len(samples)}")
    return np.array(features)


def train_svm(X_pos, X_neg, C=0.01, test_size=0.2):
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos)), -np.ones(len(X_neg))])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, ratio: {sum(y_train==1)}:{sum(y_train==-1)}")
    model = LinearSVC(C=C, max_iter=10000, dual=False, random_state=42)
    model.fit(X_train, y_train)
    print(f"  Train acc: {model.score(X_train, y_train):.4f}, Val acc: {model.score(X_val, y_val):.4f}")
    return model


def save_opencv_detector(model, output_path):
    weights = model.coef_.flatten().astype(np.float32)
    bias = model.intercept_[0].astype(np.float32)
    detector = np.append(weights, bias)
    np.savez(output_path, weights=weights, bias=bias, detector=detector)
    print(f"  Saved: {output_path} ({len(weights)} weights, bias={bias:.6f})")


def main():
    parser = argparse.ArgumentParser(description="Retrain HOG+SVM pedestrian detector")
    parser.add_argument("--videos-dir", default="/home/caizhihao/cv_project/data/custom/videos")
    parser.add_argument("--labels-base", default="/home/caizhihao/cv_project/data/custom/annotations")
    parser.add_argument("--scenes", nargs="+", default=["scene1","scene2","scene3","scene4"])
    parser.add_argument("--output", default="/home/caizhihao/traditional_method/custom_svm_weights.npz")
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--max-frames-per-scene", type=int, default=150)
    parser.add_argument("--pos-per-frame", type=int, default=8)
    parser.add_argument("--neg-per-frame", type=int, default=15)
    args = parser.parse_args()

    print("=" * 60)
    print("  HOG+SVM Custom Training")
    print("=" * 60)

    # Phase 1: Sample extraction
    print("\n[Phase 1] Sample extraction")
    all_positives, all_negatives = [], []

    for scene in args.scenes:
        vp = Path(args.videos_dir) / f"{scene}.mp4"
        ld = Path(args.labels_base) / scene / "train"
        if not vp.exists() or not ld.exists():
            print(f"  SKIP {scene}: missing video or labels")
            continue
        print(f"\n  Processing {scene}...")
        pos, neg = extract_samples(vp, ld, args.pos_per_frame, args.neg_per_frame, args.max_frames_per_scene)
        all_positives.extend(pos)
        all_negatives.extend(neg)

    print(f"\n  Total: pos={len(all_positives)}, neg={len(all_negatives)}")
    if not all_positives:
        print("ERROR: no positive samples extracted")
        sys.exit(1)

    # Balance
    n_neg = min(len(all_negatives), len(all_positives) * 3)
    np.random.shuffle(all_negatives)
    all_negatives = all_negatives[:n_neg]
    print(f"  Balanced: pos={len(all_positives)}, neg={len(all_negatives)}")

    # Phase 2: HOG features
    print(f"\n[Phase 2] HOG feature extraction")
    t0 = time.time()
    X_pos = extract_hog_features(all_positives)
    X_neg = extract_hog_features(all_negatives)
    print(f"  Done ({time.time()-t0:.1f}s)")

    # Phase 3: Train
    print(f"\n[Phase 3] SVM training (C={args.C})")
    model = train_svm(X_pos, X_neg, C=args.C)

    # Phase 4: Save
    print(f"\n[Phase 4] Save weights")
    save_opencv_detector(model, args.output)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("  Use: python traditional_method/run_traditional.py --custom-svm")
    print("=" * 60)


if __name__ == "__main__":
    main()
