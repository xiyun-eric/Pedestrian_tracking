"""
数据集统计工具

统计:
  - 图像数量
  - 标注数量
  - 标注框数量
  - 类别分布
  - 空/非空标注比例

使用方法:
  python tools/data_stats.py --data-dir data/custom
"""

import argparse
from pathlib import Path
from collections import defaultdict


CLASSES = {
    0: "pedestrian",
    1: "cyclist",
    2: "car",
    3: "other",
}


def count_files(directory: Path, pattern: str) -> int:
    """统计文件数量"""
    return len(list(directory.glob(pattern)))


def analyze_annotations(label_dir: Path) -> dict:
    """分析标注文件"""
    total_boxes = 0
    class_counts = defaultdict(int)
    empty_files = 0
    non_empty_files = 0
    
    for label_file in sorted(label_dir.glob("*.txt")):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) == 0:
            empty_files += 1
        else:
            non_empty_files += 1
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    total_boxes += 1
                    class_counts[cls_id] += 1
    
    return {
        "total_boxes": total_boxes,
        "class_counts": dict(class_counts),
        "empty_files": empty_files,
        "non_empty_files": non_empty_files,
        "total_label_files": empty_files + non_empty_files,
    }


def analyze_scene(scene_dir: Path, annotations_dir: Path) -> dict:
    """分析单个场景"""
    stats = {}
    
    for split in ["train", "val"]:
        img_split_dir = scene_dir / split
        ann_split_dir = annotations_dir / split
        
        if img_split_dir.exists():
            img_count = count_files(img_split_dir, "*.png") + count_files(img_split_dir, "*.jpg")
            stats[f"images_{split}"] = img_count
        else:
            stats[f"images_{split}"] = 0
        
        if ann_split_dir.exists():
            ann_stats = analyze_annotations(ann_split_dir)
            stats[f"annotations_{split}"] = ann_stats
        else:
            stats[f"annotations_{split}"] = None
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="数据集统计")
    parser.add_argument("--data-dir", type=str, default="data/custom",
                       help="数据目录")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    print("=" * 60)
    print("数据集统计报告")
    print("=" * 60)
    
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    
    if not images_dir.exists():
        print(f"错误: 图像目录不存在: {images_dir}")
        return
    
    total_images = 0
    total_boxes = 0
    total_class_counts = defaultdict(int)
    
    for scene_dir in sorted(images_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        scene_name = scene_dir.name
        ann_scene_dir = annotations_dir / scene_name
        
        print(f"\n场景: {scene_name}")
        print("-" * 40)
        
        scene_stats = analyze_scene(scene_dir, ann_scene_dir)
        
        for split in ["train", "val"]:
            img_count = scene_stats.get(f"images_{split}", 0)
            ann_stats = scene_stats.get(f"annotations_{split}")
            
            print(f"  {split}:")
            print(f"    图像数: {img_count}")
            
            if ann_stats:
                print(f"    标注文件数: {ann_stats['total_label_files']}")
                print(f"    有标注文件: {ann_stats['non_empty_files']}")
                print(f"    空标注文件: {ann_stats['empty_files']}")
                print(f"    标注框总数: {ann_stats['total_boxes']}")
                
                if ann_stats['class_counts']:
                    print(f"    类别分布:")
                    for cls_id, count in sorted(ann_stats['class_counts'].items()):
                        cls_name = CLASSES.get(cls_id, f"class_{cls_id}")
                        print(f"      {cls_name}: {count}")
                
                total_boxes += ann_stats['total_boxes']
                for cls_id, count in ann_stats['class_counts'].items():
                    total_class_counts[cls_id] += count
            
            total_images += img_count
    
    print("\n" + "=" * 60)
    print("总计")
    print("=" * 60)
    print(f"总图像数: {total_images}")
    print(f"总标注框数: {total_boxes}")
    print(f"类别分布:")
    for cls_id, count in sorted(total_class_counts.items()):
        cls_name = CLASSES.get(cls_id, f"class_{cls_id}")
        print(f"  {cls_name}: {count}")
    
    # 计算平均每帧框数
    if total_images > 0:
        avg_boxes = total_boxes / total_images
        print(f"平均每帧框数: {avg_boxes:.2f}")


if __name__ == "__main__":
    main()