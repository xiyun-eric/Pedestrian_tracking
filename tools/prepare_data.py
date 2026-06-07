"""
准备数据集用于 YOLO 训练

将当前数据结构转换为 ultralytics 需要的标准格式:
  - images/train/*.png
  - labels/train/*.txt

使用方法:
  python tools/prepare_data.py --input data/custom --output data/yolo_custom
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def prepare_dataset(input_dir: Path, output_dir: Path):
    """
    准备数据集
    
    输入结构:
      input_dir/images/scene1/train/*.png
      input_dir/annotations/scene1/train/*.txt
    
    输出结构:
      output_dir/images/train/*.png
      output_dir/labels/train/*.txt
    """
    input_images = input_dir / "images"
    input_annotations = input_dir / "annotations"
    
    output_images = output_dir / "images"
    output_labels = output_dir / "labels"
    
    # 创建输出目录
    for split in ["train", "val"]:
        (output_images / split).mkdir(parents=True, exist_ok=True)
        (output_labels / split).mkdir(parents=True, exist_ok=True)
    
    # 统计
    stats = {"train": {"images": 0, "labels": 0}, "val": {"images": 0, "labels": 0}}
    
    # 遍历所有场景
    for scene_dir in sorted(input_images.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        scene_name = scene_dir.name
        
        for split in ["train", "val"]:
            src_img_dir = scene_dir / split
            src_label_dir = input_annotations / scene_name / split
            
            if not src_img_dir.exists():
                continue
            
            # 复制图像
            for img_path in tqdm(
                sorted(src_img_dir.glob("*.png")) + sorted(src_img_dir.glob("*.jpg")),
                desc=f"复制 {scene_name}/{split} 图像"
            ):
                # 使用场景名作为前缀避免文件名冲突
                new_name = f"{scene_name}_{img_path.name}"
                dst_img = output_images / split / new_name
                shutil.copy2(img_path, dst_img)
                stats[split]["images"] += 1
                
                # 复制标签（如果存在）
                label_path = src_label_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    dst_label = output_labels / split / f"{scene_name}_{img_path.stem}.txt"
                    shutil.copy2(label_path, dst_label)
                    stats[split]["labels"] += 1
                else:
                    # 创建空标签文件（帧中没有人）
                    dst_label = output_labels / split / f"{scene_name}_{img_path.stem}.txt"
                    dst_label.touch()
    
    return stats


def create_dataset_yaml(output_dir: Path):
    """创建 dataset.yaml 配置文件"""
    yaml_content = f"""# 自采集行人跟踪数据集
# 用于 YOLO11 微调

# 数据集路径 (相对于 configs 目录)
path: {output_dir.absolute()}

# 训练/验证集
train: images/train
val: images/val

# 类别定义
names:
  0: pedestrian  # 行人
  1: cyclist     # 骑车人
  2: car         # 车辆
  3: other       # 其他

# 数据集统计
# 总图像数: 见下方
# 类别分布: pedestrian (82%), cyclist (12%), car (6%)
"""
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="准备数据集用于 YOLO 训练")
    parser.add_argument("--input", type=str, default="data/custom",
                       help="输入数据目录")
    parser.add_argument("--output", type=str, default="data/yolo_custom",
                       help="输出数据目录")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print(f"\n{'='*60}")
    print("准备数据集")
    print(f"{'='*60}")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在")
        return
    
    # 准备数据
    stats = prepare_dataset(input_dir, output_dir)
    
    # 创建配置文件
    yaml_path = create_dataset_yaml(output_dir)
    
    # 打印统计
    print(f"\n{'='*60}")
    print("数据准备完成")
    print(f"{'='*60}")
    print(f"训练集: {stats['train']['images']} 图像, {stats['train']['labels']} 标签")
    print(f"验证集: {stats['val']['images']} 图像, {stats['val']['labels']} 标签")
    print(f"\n配置文件: {yaml_path}")
    print(f"\n使用方法:")
    print(f"  python deep_method/model/train_yolo.py --data {yaml_path}")


if __name__ == "__main__":
    main()