# 动态场景多目标跟踪 - 项目完整教程

## 项目概述

本项目实现了**动态场景下的行人多目标跟踪**，包含完整的数据准备、模型微调、跟踪测试流程。

### 核心功能

| 功能         | 脚本                                | 说明                     |
| ---------- | --------------------------------- | ---------------------- |
| **数据准备**   | `tools/dataset_creator.py`        | 视频分帧                   |
| **AI自动标注** | `tools/annotator.py`              | YOLO预标注                |
| **人工调整**   | `tools/annotator.py`              | 手动修正标注                 |
| **数据转换**   | `tools/prepare_data.py`           | 转换为YOLO格式              |
| **模型微调**   | `deep_method/model/train_yolo.py` | YOLO11微调训练             |
| **行人跟踪**   | `run_tracking_evaluation.py`      | AdvancedTracker + ReID |

***

## 完整流程

### 流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         数据准备阶段                                  │
├─────────────────────────────────────────────────────────────────────┤
│  1. 视频分帧 → tools/dataset_creator.py                              │
│  2. AI自动标注 → tools/annotator.py --auto-label                     │
│  3. 人工调整 → tools/annotator.py (手动模式)                          │
│  4. 数据转换 → tools/prepare_data.py                                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         模型微调阶段                                  │
├─────────────────────────────────────────────────────────────────────┤
│  5. YOLO11微调 → deep_method/model/train_yolo.py                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         行人跟踪阶段                                  │
├─────────────────────────────────────────────────────────────────────┤
│  6. 跟踪测试 → test_tracking_phase3.py                               │
└─────────────────────────────────────────────────────────────────────┘
```

***

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证环境

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

***

## 阶段一：数据准备

### 步骤1：视频分帧

将原始视频提取为图像帧，并自动划分训练集/验证集。

```bash
python tools/dataset_creator.py --mode extract \
    --video data/custom/videos/scene1.mp4 \
    --output data/custom/images/scene1 \
    --target-fps 10 \
    --split-ratio 0.67
```

**参数说明：**

- `--video`: 原始视频路径
- `--output`: 输出图像目录
- `--target-fps`: 提取帧率（10fps = 每秒提取10帧）
- `--split-ratio`: 训练集比例（0.67 = 67%训练，33%验证）

**输出结构：**

```
data/custom/images/scene1/
├── train/          # 训练集图像
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
└── val/            # 验证集图像
    ├── frame_000000.png
    ├── frame_000001.png
    └── ...
```

***

### 步骤2：AI自动标注

使用YOLO模型自动生成初始标注（粗标注）。

```bash
python tools/annotator.py \
    --img-dir data/custom/images/scene1/train \
    --output data/custom/annotations/scene1/train \
    --auto-label \
    --batch
```

**参数说明：**

- `--img-dir`: 图像目录
- `--output`: 标注输出目录
- `--auto-label`: 启用AI自动标注
- `--batch`: 批量处理模式

**输出：**

```
data/custom/annotations/scene1/train/
├── frame_000000.txt   # YOLO格式标注
├── frame_000001.txt
└── ...
```

***

### 步骤3：人工调整标注

在AI预标注基础上进行人工微调修正。

```bash
python tools/annotator.py \
    --img-dir data/custom/images/scene1/train \
    --output data/custom/annotations/scene1/train \
    --auto-label
```

**操作说明：**

| 按键    | 功能                |
| ----- | ----------------- |
| `e`   | 下一帧（自动保存）         |
| `q`   | 上一帧（自动保存）         |
| `a`   | 新增模式（拖拽画新框）       |
| `s`   | 调整模式（拖拽移动/缩放已有框）  |
| `d`   | 删除模式（点击删除框）       |
| `1`   | 类别：pedestrian（行人） |
| `2`   | 类别：cyclist（骑车人）   |
| `3`   | 类别：car（车辆）        |
| `4`   | 类别：other（其他）      |
| `w`   | 保存当前标注            |
| `c`   | 清除当前帧所有标注         |
| `ESC` | 退出                |

***

### 步骤4：数据转换

将场景结构转换为YOLO标准格式。

```bash
python tools/prepare_data.py \
    --input data/custom \
    --output data/yolo_custom
```

**输入结构：**

```
data/custom/
├── images/
│   ├── scene1/train/*.png
│   ├── scene1/val/*.png
│   ├── scene2/train/*.png
│   └── scene2/val/*.png
└── annotations/
    ├── scene1/train/*.txt
    ├── scene1/val/*.txt
    ├── scene2/train/*.txt
    └── scene2/val/*.txt
```

**输出结构（YOLO标准格式）：**

```
data/yolo_custom/
├── images/
│   ├── train/
│   │   ├── scene1_frame_000000.png
│   │   ├── scene2_frame_000000.png
│   │   └── ...
│   └── val/
│       ├── scene1_frame_000000.png
│       ├── scene2_frame_000000.png
│       └── ...
├── labels/
│   ├── train/
│   │   ├── scene1_frame_000000.txt
│   │   ├── scene2_frame_000000.txt
│   │   └── ...
│   └── val/
│       ├── scene1_frame_000000.txt
│       ├── scene2_frame_000000.txt
│       └── ...
└── dataset.yaml
```

***

## 阶段二：模型微调

### 步骤5：YOLO11微调训练

使用自定义数据集微调YOLO11模型。

```bash
python deep_method/model/train_yolo.py \
    --data data/yolo_custom/dataset.yaml \
    --epochs 30 \
    --batch 4 \
    --device 0
```

**参数说明：**

| 参数           | 默认值                    | 说明             |
| ------------ | ---------------------- | -------------- |
| `--data`     | `configs/dataset.yaml` | 数据集配置文件        |
| `--model`    | `yolo11m.pt`           | 基础模型           |
| `--epochs`   | 30                     | 训练轮数           |
| `--batch`    | 4                      | 批次大小           |
| `--img-size` | 640                    | 输入尺寸           |
| `--lr`       | 0.001                  | 学习率            |
| `--device`   | auto                   | 设备（auto/cpu/0） |

**训练输出：**

```
runs/detect/weights/yolo_custom/train/
├── weights/
│   ├── best.pt      # 最佳模型
│   └── last.pt      # 最终模型
├── results.csv      # 训练指标
├── results.png      # 训练曲线
└── confusion_matrix.png
```

***

## 阶段三：行人跟踪

### 步骤6：跟踪测试

支持三种跟踪方式：**传统方法**（HOG+光流+Sort）、**深度方法-预训练模型**（YOLO11m+AdvancedTracker）、**深度方法-微调模型**（本地训练YOLO+AdvancedTracker）。

#### 6.1 传统方法跟踪

使用HOG行人检测 + 光流运动估计 + Sort跟踪，无需GPU和深度模型。

```bash
# 视频模式（默认处理全部帧）
python traditional_method/run_traditional.py \
    --video data/custom/videos/scene1.mp4

# 限制帧数
python traditional_method/run_traditional.py \
    --video data/custom/videos/scene1.mp4 \
    --frames 200

# 禁用光流或ReID
python traditional_method/run_traditional.py \
    --video data/custom/videos/scene1.mp4 \
    --no-optical-flow
python traditional_method/run_traditional.py \
    --video data/custom/videos/scene1.mp4 \
    --no-reid

# 启用GT评估
python traditional_method/run_traditional.py \
    --video data/custom/videos/scene1.mp4 \
    --eval --labels data/yolo_custom/labels/val --scene scene1
```

**参数说明：**

| 参数                  | 默认值  | 说明           |
| ------------------- | ---- | ------------ |
| `--video`           | -    | 视频文件路径       |
| `--frames`          | 全部帧  | 最大处理帧数       |
| `--hog-conf`        | 0.3  | HOG检测置信度阈值   |
| `--hog-scale`       | 1.05 | HOG图像金字塔缩放系数 |
| `--max-age`         | 30   | 轨迹最大丢失帧数     |
| `--min-hits`        | 3    | 确认轨迹所需最小检测数  |
| `--iou-threshold`   | 0.3  | IoU匹配阈值      |
| `--no-optical-flow` | -    | 禁用光流运动估计     |
| `--no-reid`         | -    | 禁用ReID重识别    |

#### 6.2 深度方法跟踪（预训练模型）

使用预训练YOLO11m模型 + AdvancedTracker + ReID，无需微调即可运行。

```bash
# 默认处理全部帧
python run_tracking_evaluation.py \
    --video data/custom/videos/scene1.mp4 \
    --model pretrained

# 限制帧数
python run_tracking_evaluation.py \
    --video data/custom/videos/scene1.mp4 \
    --frames 100 \
    --model pretrained

# 启用GT评估
python run_tracking_evaluation.py \
    --video data/custom/videos/scene1.mp4 \
    --model pretrained \
    --eval --labels data/yolo_custom/labels/val --scene scene1
```

#### 6.3 深度方法跟踪（微调模型）

使用本地训练的YOLO模型 + AdvancedTracker + ReID，精度最高。

```bash
# 默认处理全部帧
python run_tracking_evaluation.py \
    --video data/custom/videos/scene1.mp4 \
    --model custom

# 限制帧数
python run_tracking_evaluation.py \
    --video data/custom/videos/scene1.mp4 \
    --frames 100 \
    --model custom

# 启用GT评估
python run_tracking_evaluation.py \
    --video data/custom/videos/scene1.mp4 \
    --model custom \
    --eval --labels data/yolo_custom/labels/val --scene scene1
```

**深度方法参数说明：**

| 参数         | 默认值                             | 说明                      |
| ---------- | ------------------------------- | ----------------------- |
| `--video`  | `data/custom/videos/scene1.mp4` | 测试视频                    |
| `--frames` | 0                               | 处理帧数，0表示处理全部帧           |
| `--model`  | custom                          | 模型类型（custom/pretrained） |
| `--eval`   | -                               | 启用GT评估                  |
| `--labels` | `data/yolo_custom/labels/val`   | YOLO格式GT标注目录            |
| `--scene`  | -                               | 场景名称过滤（如scene1, scene2） |

**跟踪输出：**

```
tracking_evaluation/scene1/custom/
├── tracking_result.mp4       # 跟踪结果视频
├── evaluation_report.json    # 评估报告（--eval时生成）
├── evaluation_report.csv     # 评估CSV（--eval时生成）
├── predictions.txt           # 预测结果MOT格式
└── gt.txt                    # GT标注MOT格式
```

***

## KITTI 数据集跟踪

本项目支持对 KITTI 行人跟踪数据集的图像序列进行检测和跟踪。

### 数据集结构

```
data/kitti/
├── 0017/              # KITTI 序列 0017
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── 0019/              # KITTI 序列 0019
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── labels/            # YOLO 格式 GT 标注（可选）
    ├── 0017_000000.txt
    └── ...
```

### 传统方法跟踪

```bash
# 单个序列
python traditional_method/run_traditional.py --data-dir data/kitti --seq 0017

# 多个序列
python traditional_method/run_traditional.py --data-dir data/kitti --seq 0017 0019

# 限制帧数
python traditional_method/run_traditional.py --data-dir data/kitti --seq 0017 --frames 100
```

### 深度方法跟踪

```bash
# 使用微调模型
python run_tracking_evaluation.py --images data/kitti/0017 --model custom --frames 200

# 使用预训练模型
python run_tracking_evaluation.py --images data/kitti/0017 --model pretrained --frames 200
```

### 对比实验

```bash
# KITTI 数据集一键对比（传统方法 + 深度方法）
python run_comparison.py --kitti --kitti-seq 0017 0019 --frames 200

# 仅对比特定方法
python run_comparison.py --kitti --kitti-seq 0017 --methods traditional deep_pretrained
```

**参数说明：**

| 参数            | 默认值          | 说明             |
| ------------- | ------------ | -------------- |
| `--kitti`     | -            | 启用 KITTI 数据集模式 |
| `--kitti-dir` | `data/kitti` | KITTI 图像目录     |
| `--kitti-seq` | `0017 0019`  | 要处理的序列名称       |
| `--images`    | -            | 图像序列目录（深度方法）   |
| `--data-dir`  | -            | 图像根目录（传统方法）    |
| `--seq`       | -            | 序列名称（传统方法）     |

***

## 项目结构

```
cv_project/
├── data/                          # 数据集
│   ├── custom/                    # 自主采集数据集（原始）
│   │   ├── videos/                # 原始视频
│   │   ├── images/                # 提取的图像帧（场景结构）
│   │   └── annotations/           # 标注文件
│   └── yolo_custom/               # YOLO格式数据集（转换后）
│       ├── images/train/          # 训练图像
│       ├── images/val/            # 验证图像
│       ├── labels/train/          # 训练标签
│       ├── labels/val/            # 验证标签
│       └── dataset.yaml           # 数据集配置
│   ├── kitti/                    # KITTI 数据集
│   │   ├── 0017/                 # 序列 0017 图像
│   │   ├── 0019/                 # 序列 0019 图像
│   │   └── labels/               # YOLO 格式 GT 标注
│
├── deep_method/                   # 深度学习方法
│   ├── detector.py               # YOLO检测器封装
│   ├── tracking/                 # 跟踪模块
│   │   ├── advanced_tracker.py   # AdvancedTracker
│   │   ├── reid_extractor.py     # ReID特征提取
│   │   ├── kalman_filter.py      # 卡尔曼滤波
│   │   └── tracking_config.py    # 配置预设
│   └── model/
│       ├── train_yolo.py         # YOLO微调训练
│       ├── train_lora.py         # LoRA微调（可选）
│       ├── custom_loss.py        # 自定义损失
│       └── lora_layers.py        # LoRA层
│
├── tools/                         # 工具集
│   ├── dataset_creator.py        # 视频采集+分帧
│   ├── annotator.py              # 标注工具（AI+人工）
│   ├── prepare_data.py           # 数据转换
│   ├── data_stats.py             # 数据统计
│   └── evaluate.py               # 评估指标
│
├── runs/                          # 训练输出
│   └── detect/weights/yolo_custom/train/
│       └── weights/best.pt       # 微调后的模型
│
├── test_tracking_phase3.py        # 跟踪测试脚本
├── configs/dataset.yaml           # 数据集配置
├── requirements.txt               # 依赖列表
└── docs/
    ├── README.md                  # 本文档
    ├── ID_SWITCH_SOLUTIONS.md     # ID切换问题解决方案
    └── TASK_PLAN.md               # 任务计划
```

***

## 快速开始（完整流程）

### 从零开始的完整流程

```bash
# 1. 视频分帧
python tools/dataset_creator.py --mode extract \
    --video data/custom/videos/scene1.mp4 \
    --output data/custom/images/scene1 \
    --target-fps 10 --split-ratio 0.67

# 2. AI自动标注
python tools/annotator.py \
    --img-dir data/custom/images/scene1/train \
    --output data/custom/annotations/scene1/train \
    --auto-label --batch

# 3. 人工调整（可选）
python tools/annotator.py \
    --img-dir data/custom/images/scene1/train \
    --output data/custom/annotations/scene1/train \
    --auto-label

# 4. 数据转换
python tools/prepare_data.py \
    --input data/custom \
    --output data/yolo_custom

# 5. 模型微调
python deep_method/model/train_yolo.py \
    --data data/yolo_custom/dataset.yaml \
    --epochs 30 --batch 4

# 6. 跟踪测试
python test_tracking_phase3.py \
    --video data/custom/videos/scene1.mp4 \
    --frames 100 --model custom
```

### 使用已有数据快速测试

```bash
# 如果已有微调模型，直接运行跟踪测试
python run_tracking_evaluation.py \
    --video data/custom/videos/scene1.mp4 \
    --model custom

# 如果没有微调模型，使用预训练模型
python run_tracking_evaluation.py \
    --video data/custom/videos/scene1.mp4 \
    --model pretrained
```

***

## 跟踪算法说明

### AdvancedTracker 特性

| 特性            | 说明               |
| ------------- | ---------------- |
| **ByteTrack** | 低分框二次匹配，减少漏检     |
| **ReID**      | OSNet外观特征，稳定ID   |
| **卡尔曼滤波**     | 运动预测，处理遮挡        |
| **EMA平滑**     | 特征历史权重70%，减少ID切换 |
| **社会约束**      | 避免轨迹交叉           |

### 配置预设

| 预设              | max\_age | min\_hits | feature\_smooth\_alpha | 适用场景 |
| --------------- | -------- | --------- | ---------------------- | ---- |
| standard        | 30       | 3         | 0.3                    | 一般场景 |
| fast            | 15       | 2         | 0.5                    | 快速处理 |
| high\_precision | 50       | 5         | 0.2                    | 高精度  |
| crowded\_scene  | 20       | 4         | 0.3                    | 担忧场景 |

***

## 常见问题

### 1. CUDA out of memory

```bash
# 减小batch size
python deep_method/model/train_yolo.py --batch 2

# 使用CPU
python deep_method/model/train_yolo.py --device cpu
```

### 2. 标注工具无界面

需要在带显示器的终端运行（WSL2需配置X11转发）。

### 3. 模型文件不存在

```bash
# 检查模型路径
ls runs/detect/weights/yolo_custom/train/weights/best.pt

# 如果不存在，使用预训练模型
python run_tracking_evaluation.py --model pretrained
```

### 4. ID切换频繁

参考 [ID\_SWITCH\_SOLUTIONS.md](ID_SWITCH_SOLUTIONS.md) 了解解决方案。

***

## 技术细节

### 卡尔曼滤波

- 8维状态: \[x, y, w, h, vx, vy, vw, vh]
- 匀速运动模型
- 观测: \[x, y, w, h]

### ByteTrack二次匹配

- 高分框（conf >= 0.5）：首次匹配
- 低分框（conf < 0.5）：二次匹配丢失轨迹

### ReID特征提取

- OSNet x1\_0（MSMT17预训练）
- EMA平滑：alpha=0.3（历史70%，当前30%）

### 代价矩阵

- 外观距离（ReID特征）
- IoU距离（空间重叠）
- 马氏距离（运动预测）
- 加权融合：appearance\_weight + iou\_weight + mahal\_weight

