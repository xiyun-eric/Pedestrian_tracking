# convert_kitti_labels.py
from code.filter_code.kitti_tracking_pipeline import process_labels_to_yolo
from pathlib import Path

# 转换 KITTI 标签为 YOLO 格式
process_labels_to_yolo(
    Path('data/kitti/raw_labels/training/label_02'),  # 原始标签路径
    Path('data/kitti/labels'),                        # 输出 YOLO 标签路径
    {'Pedestrian', 'Person_sitting', 'Cyclist'}       # 只保留这三类目标
    
)
