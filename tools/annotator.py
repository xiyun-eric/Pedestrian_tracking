"""
标注工具 (基于 OpenCV GUI)

功能:
  1. YOLO 自动预标注 (粗标注)
  2. 人工微调 (在预标注基础上调整)
  3. 三种操作模式: 新增 / 调整 / 删除
  4. 类别选择: pedestrian / cyclist / car / other
  5. 数字键跳转指定图片

操作说明:
  翻页:
    - 'e': 下一帧 (自动保存)
    - 'q': 上一帧 (自动保存)
    - 'g' + 数字 + Enter: 跳转到指定编号图片

  模式切换:
    - 'a': 新增模式 (拖拽画新框)
    - 's': 调整模式 (拖拽移动/缩放已有框)
    - 'd': 删除模式 (点击删除框)

  类别切换 (新增/调整模式下):
    - '1': pedestrian (行人) - 绿色
    - '2': cyclist (骑车人) - 橙色
    - '3': car (车辆) - 红色
    - '4': other (其他) - 灰色

  其他:
    - 'w': 保存当前标注
    - 'c': 清除当前帧所有标注
    - 'ESC': 退出

使用方法:
  # 仅人工标注
  python tools/annotator.py --img-dir data/custom/images/scene1/train --output data/custom/annotations/scene1/train

  # YOLO 预标注 + 人工微调
  python tools/annotator.py --img-dir data/custom/images/scene1/train --output data/custom/annotations/scene1/train --auto-label

  # 指定 YOLO 模型
  python tools/annotator.py --img-dir data/custom/images/scene1/train --output data/custom/annotations/scene1/train --auto-label --model yolo11n.pt
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
import argparse
from typing import List, Tuple, Optional


CLASSES = {
    0: ("pedestrian", (0, 255, 0)),
    1: ("cyclist", (255, 165, 0)),
    2: ("car", (0, 0, 255)),
    3: ("other", (128, 128, 128)),
}

MODE_ADD = 0
MODE_ADJUST = 1
MODE_DELETE = 2
MODE_NAMES = {MODE_ADD: "ADD", MODE_ADJUST: "ADJUST", MODE_DELETE: "DELETE"}

HANDLE_SIZE = 8
MIN_BOX_SIZE = 5


def yolo_to_xyxy(cx: float, cy: float, bw: float, bh: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


class YOLOAutoLabeler:
    """YOLO 自动预标注"""

    def __init__(self, model_name: str = "yolo11n.pt", conf: float = 0.25, device: str = "cuda"):
        self.model = None
        self.model_name = model_name
        self.conf = conf
        self.device = device
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            print(f"YOLO 模型加载成功: {self.model_name}")
        except Exception as e:
            print(f"YOLO 模型加载失败: {e}")
            print("将跳过自动预标注")
            self.model = None

    def detect(self, image: np.ndarray) -> List[dict]:
        if self.model is None:
            return []

        results = self.model(image, conf=self.conf, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                yolo_cls = self._map_class(cls_id)
                if yolo_cls is not None:
                    detections.append({
                        "class_id": yolo_cls,
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "confidence": conf,
                    })

        return detections

    def _map_class(self, coco_cls_id: int) -> Optional[int]:
        mapping = {
            0: 0,   # person → pedestrian
            1: 1,   # bicycle → cyclist
            2: 2,   # car → car
            3: 1,   # motorcycle → cyclist
            5: 2,   # bus → car
            7: 2,   # truck → car
        }
        return mapping.get(coco_cls_id, None)


class AnnotatorTool:
    """OpenCV 标注工具"""

    def __init__(
        self,
        img_dir: Path,
        output_dir: Path,
        auto_labeler: Optional[YOLOAutoLabeler] = None,
    ):
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.auto_labeler = auto_labeler

        self.images = sorted(
            list(img_dir.glob("*.png")) +
            list(img_dir.glob("*.jpg")) +
            list(img_dir.glob("*.jpeg"))
        )

        if not self.images:
            raise ValueError(f"未在 {img_dir} 中找到图像")

        self.current_idx = 0
        self.current_class = 0
        self.current_mode = MODE_ADD

        self.annotations: List[dict] = []
        self.selected_idx: Optional[int] = None
        self.drawing = False
        self.start_point: Optional[Tuple[int, int]] = None
        self.end_point: Optional[Tuple[int, int]] = None
        self.drag_offset: Optional[Tuple[int, int]] = None
        self.resize_handle: Optional[str] = None
        self.resize_origin: Optional[dict] = None

        self.jump_input = ""
        self.jump_mode = False

        self._load_existing_annotations()

    def run(self):
        print(f"\n标注工具启动")
        print(f"图像目录: {self.img_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"总帧数: {len(self.images)}")
        if self.auto_labeler:
            print(f"自动预标注: 已启用")
        print(f"\n操作说明:")
        print(f"  'e': 下一帧  'q': 上一帧")
        print(f"  'a': 新增模式  's': 调整模式  'd': 删除模式")
        print(f"  '1-4': 切换类别 (pedestrian/cyclist/car/other)")
        print(f"  'g': 输入编号跳转图片")
        print(f"  'w': 保存  'c': 清除当前帧标注  'ESC': 退出")

        cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotator", self._mouse_callback)

        while True:
            frame = self._load_current_frame()
            if frame is None:
                break

            display = frame.copy()
            self._draw_annotations(display)

            if self.drawing and self.start_point and self.end_point and self.current_mode == MODE_ADD:
                color = CLASSES[self.current_class][1]
                cv2.rectangle(display, self.start_point, self.end_point, color, 2)

            self._draw_info(display)

            if self.jump_mode:
                h, w = display.shape[:2]
                overlay = display.copy()
                cv2.rectangle(overlay, (w // 2 - 150, h // 2 - 30), (w // 2 + 150, h // 2 + 30), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
                cv2.putText(display, f"Jump to: {self.jump_input}_", (w // 2 - 130, h // 2 + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            cv2.imshow("Annotator", display)

            key = cv2.waitKey(1) & 0xFF

            if self.jump_mode:
                self._handle_jump_input(key)
                continue

            if key == 27:
                self._save_current()
                break
            elif key == ord('e'):
                self._save_current()
                self.current_idx = min(self.current_idx + 1, len(self.images) - 1)
                self._load_existing_annotations()
            elif key == ord('q'):
                self._save_current()
                self.current_idx = max(self.current_idx - 1, 0)
                self._load_existing_annotations()
            elif key == ord('w'):
                self._save_current()
                print(f"已保存: {self.images[self.current_idx].name}")
            elif key == ord('c'):
                self.annotations = []
                self.selected_idx = None
                print("已清除当前帧标注")
            elif key == ord('a'):
                self.current_mode = MODE_ADD
                self.selected_idx = None
                print(f"模式: 新增 (ADD)")
            elif key == ord('s'):
                self.current_mode = MODE_ADJUST
                print(f"模式: 调整 (ADJUST)")
            elif key == ord('d'):
                self.current_mode = MODE_DELETE
                self.selected_idx = None
                print(f"模式: 删除 (DELETE)")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                cls_id = int(chr(key)) - 1  # 按键1-4对应类别0-3
                if cls_id in CLASSES:
                    self.current_class = cls_id
                    if self.selected_idx is not None and self.current_mode == MODE_ADJUST:
                        self.annotations[self.selected_idx]["class_id"] = cls_id
                        print(f"框 {self.selected_idx} 类别改为: {CLASSES[cls_id][0]}")
                    else:
                        print(f"当前类别: {CLASSES[self.current_class][0]}")
            elif key == ord('g'):
                self.jump_mode = True
                self.jump_input = ""

        cv2.destroyAllWindows()
        print(f"\n标注完成! 共标注 {self._count_total_annotations()} 个框")

    def _handle_jump_input(self, key):
        if key == 13:  # Enter
            if self.jump_input.isdigit():
                target = int(self.jump_input) - 1
                if 0 <= target < len(self.images):
                    self._save_current()
                    self.current_idx = target
                    self._load_existing_annotations()
                    print(f"跳转到第 {target + 1} 帧")
                else:
                    print(f"无效编号: {target + 1} (范围 1-{len(self.images)})")
            self.jump_mode = False
            self.jump_input = ""
        elif key == 27:  # ESC cancel
            self.jump_mode = False
            self.jump_input = ""
        elif key == 8:  # Backspace
            self.jump_input = self.jump_input[:-1]
        elif ord('0') <= key <= ord('9'):
            self.jump_input += chr(key)

    def _mouse_callback(self, event, x, y, flags, param):
        if self.current_mode == MODE_ADD:
            self._mouse_add(event, x, y)
        elif self.current_mode == MODE_ADJUST:
            self._mouse_adjust(event, x, y)
        elif self.current_mode == MODE_DELETE:
            self._mouse_delete(event, x, y)

    def _mouse_add(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            if x2 - x1 > MIN_BOX_SIZE and y2 - y1 > MIN_BOX_SIZE:
                self.annotations.append({
                    "class_id": self.current_class,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                })

    def _mouse_adjust(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            handle = self._hit_handle(x, y)
            if handle and self.selected_idx is not None:
                self.resize_handle = handle
                ann = self.annotations[self.selected_idx]
                self.resize_origin = {"x1": ann["x1"], "y1": ann["y1"], "x2": ann["x2"], "y2": ann["y2"]}
                self.drawing = True
                return

            hit_idx = self._hit_box(x, y)
            if hit_idx is not None:
                self.selected_idx = hit_idx
                ann = self.annotations[hit_idx]
                self.drag_offset = (x - ann["x1"], y - ann["y1"])
                self.drawing = True
            else:
                self.selected_idx = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.selected_idx is None:
                return
            ann = self.annotations[self.selected_idx]
            if self.resize_handle:
                self._apply_resize(ann, x, y)
            elif self.drag_offset:
                bw = ann["x2"] - ann["x1"]
                bh = ann["y2"] - ann["y1"]
                ann["x1"] = x - self.drag_offset[0]
                ann["y1"] = y - self.drag_offset[1]
                ann["x2"] = ann["x1"] + bw
                ann["y2"] = ann["y1"] + bh

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.drag_offset = None
            self.resize_handle = None
            self.resize_origin = None

    def _mouse_delete(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            hit_idx = self._hit_box(x, y)
            if hit_idx is not None:
                removed = self.annotations.pop(hit_idx)
                print(f"删除标注: {CLASSES[removed['class_id']][0]}")

    def _hit_box(self, x: int, y: int) -> Optional[int]:
        for i in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[i]
            if ann["x1"] <= x <= ann["x2"] and ann["y1"] <= y <= ann["y2"]:
                return i
        return None

    def _hit_handle(self, x: int, y: int) -> Optional[str]:
        if self.selected_idx is None:
            return None
        ann = self.annotations[self.selected_idx]
        corners = {
            "tl": (ann["x1"], ann["y1"]),
            "tr": (ann["x2"], ann["y1"]),
            "bl": (ann["x1"], ann["y2"]),
            "br": (ann["x2"], ann["y2"]),
        }
        for name, (cx, cy) in corners.items():
            if abs(x - cx) <= HANDLE_SIZE and abs(y - cy) <= HANDLE_SIZE:
                return name
        return None

    def _apply_resize(self, ann: dict, x: int, y: int):
        h = self.resize_handle
        if "l" in h:
            ann["x1"] = min(x, ann["x2"] - MIN_BOX_SIZE)
        if "r" in h:
            ann["x2"] = max(x, ann["x1"] + MIN_BOX_SIZE)
        if "t" in h:
            ann["y1"] = min(y, ann["y2"] - MIN_BOX_SIZE)
        if "b" in h:
            ann["y2"] = max(y, ann["y1"] + MIN_BOX_SIZE)

    def _load_current_frame(self) -> Optional[np.ndarray]:
        if self.current_idx >= len(self.images):
            return None
        img_path = self.images[self.current_idx]
        frame = cv2.imread(str(img_path))
        return frame

    def _draw_annotations(self, image: np.ndarray):
        for i, ann in enumerate(self.annotations):
            cls_id = ann["class_id"]
            color = CLASSES.get(cls_id, (128, 128, 128))[1]
            class_name = CLASSES.get(cls_id, ("?", (0, 0, 0)))[0]

            thickness = 3 if i == self.selected_idx else 2
            cv2.rectangle(image, (ann["x1"], ann["y1"]), (ann["x2"], ann["y2"]), color, thickness)

            label = f"{class_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (ann["x1"], ann["y1"] - th - 6), (ann["x1"] + tw + 4, ann["y1"]), color, -1)
            cv2.putText(image, label, (ann["x1"] + 2, ann["y1"] - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if i == self.selected_idx and self.current_mode == MODE_ADJUST:
                for cx, cy in [(ann["x1"], ann["y1"]), (ann["x2"], ann["y1"]),
                               (ann["x1"], ann["y2"]), (ann["x2"], ann["y2"])]:
                    cv2.rectangle(image,
                                 (cx - HANDLE_SIZE, cy - HANDLE_SIZE),
                                 (cx + HANDLE_SIZE, cy + HANDLE_SIZE),
                                 (255, 255, 255), -1)
                    cv2.rectangle(image,
                                 (cx - HANDLE_SIZE, cy - HANDLE_SIZE),
                                 (cx + HANDLE_SIZE, cy + HANDLE_SIZE),
                                 color, 2)

    def _draw_info(self, image: np.ndarray):
        h, w = image.shape[:2]

        mode_color = {
            MODE_ADD: (0, 255, 0),
            MODE_ADJUST: (255, 255, 0),
            MODE_DELETE: (0, 0, 255),
        }

        info_lines = [
            (f"Frame: {self.current_idx + 1}/{len(self.images)}", (255, 255, 255)),
            (f"Mode: {MODE_NAMES[self.current_mode]}", mode_color[self.current_mode]),
            (f"Class: {CLASSES[self.current_class][0]}", CLASSES[self.current_class][1]),
            (f"Boxes: {len(self.annotations)}", (255, 255, 255)),
        ]

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (250, 25 * len(info_lines) + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        for i, (text, color) in enumerate(info_lines):
            cv2.putText(image, text, (10, 20 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if self.selected_idx is not None and self.current_mode == MODE_ADJUST:
            ann = self.annotations[self.selected_idx]
            cls_name = CLASSES.get(ann["class_id"], ("?",))[0]
            cv2.putText(image, f"Selected: #{self.selected_idx} {cls_name}",
                       (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def _load_existing_annotations(self):
        self.annotations = []
        self.selected_idx = None

        img_path = self.images[self.current_idx]
        label_path = self.output_dir / f"{img_path.stem}.txt"

        if label_path.exists():
            frame = cv2.imread(str(img_path))
            h, w = frame.shape[:2] if frame is not None else (375, 1242)

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x1, y1, x2, y2 = yolo_to_xyxy(
                            float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]), w, h)
                        self.annotations.append({
                            "class_id": cls_id,
                            "x1": x1, "y1": y1,
                            "x2": x2, "y2": y2,
                        })
        elif self.auto_labeler is not None:
            frame = cv2.imread(str(img_path))
            if frame is not None:
                detections = self.auto_labeler.detect(frame)
                self.annotations = detections
                if detections:
                    print(f"  自动预标注: {len(detections)} 个目标")

    def _save_current(self):
        img_path = self.images[self.current_idx]
        frame = cv2.imread(str(img_path))
        if frame is None:
            return

        h, w = frame.shape[:2]
        label_path = self.output_dir / f"{img_path.stem}.txt"

        if not self.annotations:
            if label_path.exists():
                label_path.unlink()
            return

        with open(label_path, 'w') as f:
            for ann in self.annotations:
                x1 = max(0, ann["x1"])
                y1 = max(0, ann["y1"])
                x2 = min(w, ann["x2"])
                y2 = min(h, ann["y2"])
                cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                f.write(f"{ann['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def _count_total_annotations(self) -> int:
        total = 0
        for label_path in self.output_dir.glob("*.txt"):
            with open(label_path, 'r') as f:
                total += sum(1 for _ in f)
        return total


def auto_label_batch(img_dir: Path, output_dir: Path, model_name: str = "yolo11n.pt", conf: float = 0.25):
    """批量自动预标注"""
    from ultralytics import YOLO

    model = YOLO(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(img_dir.glob("*.png")) +
        list(img_dir.glob("*.jpg")) +
        list(img_dir.glob("*.jpeg"))
    )

    if not images:
        print(f"未在 {img_dir} 中找到图像")
        return

    print(f"批量预标注: {len(images)} 张图像, 模型: {model_name}")

    class_mapping = {
        0: 0,   # person → pedestrian
        1: 1,   # bicycle → cyclist
        2: 2,   # car → car
        3: 1,   # motorcycle → cyclist
        5: 2,   # bus → car
        7: 2,   # truck → car
    }

    total_dets = 0
    for i, img_path in enumerate(images):
        label_path = output_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
        results = model(frame, conf=conf, verbose=False)

        dets = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0])
                yolo_cls = class_mapping.get(cls_id)
                if yolo_cls is not None:
                    dets.append((yolo_cls, x1, y1, x2, y2))

        if dets:
            with open(label_path, 'w') as f:
                for yolo_cls, x1, y1, x2, y2 in dets:
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                    f.write(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        total_dets += len(dets)
        if (i + 1) % 50 == 0:
            print(f"  处理进度: {i + 1}/{len(images)}, 累计检测: {total_dets}")

    print(f"批量预标注完成! 共 {total_dets} 个检测框")


def main():
    parser = argparse.ArgumentParser(description="标注工具")
    parser.add_argument("--img-dir", type=str, required=True,
                       help="图像目录")
    parser.add_argument("--output", type=str, required=True,
                       help="标注输出目录")
    parser.add_argument("--auto-label", action="store_true",
                       help="启用 YOLO 自动预标注")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                       help="YOLO 模型文件 (默认 yolo11n.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="YOLO 检测置信度阈值 (默认 0.25)")
    parser.add_argument("--batch", action="store_true",
                       help="批量预标注模式 (不启动 GUI)")

    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    output_dir = Path(args.output)

    if not img_dir.exists():
        print(f"错误: 图像目录不存在: {img_dir}")
        return

    if args.batch:
        auto_label_batch(img_dir, output_dir, args.model, args.conf)
        return

    auto_labeler = None
    if args.auto_label:
        auto_labeler = YOLOAutoLabeler(model_name=args.model, conf=args.conf)

    tool = AnnotatorTool(img_dir, output_dir, auto_labeler=auto_labeler)
    tool.run()


if __name__ == "__main__":
    main()
