# 动态场景多目标跟踪 - 项目完整教程

## 项目概述

本项目实现了**动态场景下的行人多目标跟踪**，包含完整的数据准备、模型微调、跟踪测试流程。

<br />

同时由于github上传体积限制，小组自行采集的数据集只能上传标注内容，具体视频数据文件保存在百度网盘中，其存放路径应当为data\custom\videos

通过网盘分享的文件：videos
链接: <https://pan.baidu.com/s/1KUDTOXmuhJZQhEY5DYMreQ> 提取码: eneq

公共数据集MOT17:<https://opendatalab.com/OpenDataLab/MOT17/tree/main/raw>

***

## 一、项目框架

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         数据层                                       │
├─────────────────────────────────────────────────────────────────────┤
│  自采集数据集 (scene1~5)  │  MOT17数据集                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         检测层                                       │
├─────────────────────────────────────────────────────────────────────┤
│  传统方法: HOG+SVM  │  深度方法: YOLO11+LoRA微调  │  大模型方法: Qwen2-VL+LoRA │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         跟踪层                                       │
├─────────────────────────────────────────────────────────────────────┤
│  传统方法: 光流+卡尔曼+级联匹配  │  深度方法: ByteTrack+ReID  │  大模型方法: IoU后处理跟踪 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         评估层                                       │
├─────────────────────────────────────────────────────────────────────┤
│  MOTA / MOTP / IDF1 / IDSW / Precision / Recall / MT / ML / Frag   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心功能

| 功能         | 脚本                                | 说明                     |
| ---------- | --------------------------------- | ---------------------- |
| **数据准备**   | `tools/dataset_creator.py`        | 视频分帧                   |
| **AI自动标注** | `tools/annotator.py`              | YOLO预标注                |
| **人工调整**   | `tools/annotator.py`              | 手动修正标注                 |
| **数据转换**   | `tools/prepare_data.py`           | 转换为YOLO格式              |
| **模型微调**   | `deep_method/model/train_lora.py` | YOLO11微调训练             |
| **传统方法跟踪** | `traditional_method/run_traditional.py` | HOG+SVM+卡尔曼 |
| **深度方法跟踪** | `run_tracking_evaluation.py`      | YOLO+AdvancedTracker+ReID |
| **大模型SFT数据构建** | `model_method/build_sft_data.py` | 大模型SFT数据构建 |
| **大模型LoRA微调** | `model_method/train.py` | Qwen2-VL LoRA微调训练 |
| **大模型推理可视化** | `model_method/visualize.py` | 大模型推理可视化+IoU评测 |
| **大模型MOT评估** | `model_method/evaluate.py` | 大模型MOT标准评估 |
| **对比评估** | `run_comparison.py`      | 三种方法对比评估 (传统/深度/大模型) |

### 1.3 项目结构

```
cv_project/
├── data/                          # 数据集
│   ├── custom/                    # 自主采集数据集（原始）
│   │   ├── videos/                # 原始视频
│   │   ├── images/                # 提取的图像帧（场景结构）
│   │   └── annotations/           # 标注文件
│   ├── yolo_custom/               # YOLO格式数据集（转换后）
│   │   ├── images/train/          # 训练图像
│   │   ├── images/val/            # 验证图像
│   │   ├── labels/train/          # 训练标签
│   │   ├── labels/val/            # 验证标签
│   │   └── dataset.yaml           # 数据集配置
│   └── MOT17/                     # MOT17 数据集
│
├── traditional_method/            # 传统方法
│   ├── hog_detector.py            # HOG+SVM行人检测（支持ROI裁剪+输入缩放）
│   ├── kalman_filter.py           # 卡尔曼滤波
│   ├── tracker.py                 # 轨迹管理
│   ├── tracking_pipeline.py       # 跟踪管道（支持跳帧检测+精细参数控制）
│   ├── retrain_svm.py             # SVM自定义重训练
│   └── run_traditional.py         # 运行脚本（支持速度优化参数）
│
├── deep_method/                   # 深度学习方法
│   ├── detector.py                # YOLO检测器封装
│   ├── tracking/                  # 跟踪模块
│   │   ├── advanced_tracker.py    # AdvancedTracker
│   │   ├── reid_extractor.py      # ReID特征提取
│   │   └── tracking_config.py     # 配置预设
│   └── model/
│       ├── train_lora.py          # YOLO LoRA微调
│       └── train_osnet_lora.py    # OSNet LoRA微调
│
├── model_method/                   # 大模型方法 (Qwen2-VL + LoRA)
│   ├── build_sft_data.py           # SFT训练数据构建
│   ├── dataset.py                  # SFT Dataset类
│   ├── lora_config.py              # LoRA配置
│   ├── losses.py                   # 自定义损失函数
│   ├── rl_reward.py                # RL奖励函数
│   ├── tracking_trainer.py         # 自定义Trainer
│   ├── train.py                    # LoRA微调训练入口
│   ├── visualize.py                # 推理可视化+IoU评测
│   ├── evaluate.py                 # MOT标准评估
│   └── ablation.py                 # 消融实验
│
├── tools/                         # 工具集
│   ├── dataset_creator.py         # 视频采集+分帧
│   ├── annotator.py               # 标注工具
│   ├── prepare_data.py            # 数据转换
│   └── evaluate.py                # 评估指标
│
├── run_tracking_evaluation.py     # 深度方法跟踪脚本
├── run_comparison.py              # 对比评估脚本
├── outputs/                       # 输出目录 (可视化/对比结果)
├── configs/dataset.yaml           # 数据集配置
└── docs/
    ├── README.md                  # 本文档
    └── PRESENTATION_OUTLINE.md    # 汇报大纲
```

***

## 二、环境配置

### 2.1 安装依赖

```bash
pip install -r requirements.txt
```

### 2.2 验证环境

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

***

## 三、命令行使用指南

### 3.1 数据准备

#### 步骤1：视频分帧

```bash
python tools/dataset_creator.py --mode extract \
    --video data/custom/videos/scene1.mp4 \
    --output data/custom/images/scene1 \
    --target-fps 10 \
    --split-ratio 0.67
```

| 参数            | 默认值  | 说明                |
| ------------- | ---- | ----------------- |
| `--video`     | -    | 原始视频路径            |
| `--output`    | -    | 输出图像目录            |
| `--target-fps`| 10   | 提取帧率（10fps = 每秒10帧）|
| `--split-ratio`| 0.67 | 训练集比例（67%训练，33%验证）|

#### 步骤2：AI自动标注

```bash
python tools/annotator.py \
    --img-dir data/custom/images/scene1/train \
    --output data/custom/annotations/scene1/train \
    --auto-label \
    --batch
```

#### 步骤3：数据转换

```bash
python tools/prepare_data.py \
    --input data/custom \
    --output data/yolo_custom
```

### 3.2 模型微调

#### YOLO11 LoRA 微调

```bash
python deep_method/model/train_lora.py --data configs/dataset.yaml --epochs 30 --batch 4
```

| 参数             | 默认值                    | 说明             |
| -------------- | ---------------------- | -------------- |
| `--data`       | `configs/dataset.yaml` | 数据集配置文件        |
| `--model`      | `yolo11m.pt`           | 基础模型           |
| `--epochs`     | 30                     | 训练轮数           |
| `--batch`      | 4                      | 批次大小           |
| `--r`          | 8                      | LoRA 秩         |
| `--lora-alpha` | 16.0                   | LoRA 缩放系数      |

### 3.3 传统方法跟踪

```bash
# 默认使用OpenCV API（速度快）
python traditional_method/run_traditional.py --video data/custom/videos/scene1.mp4

# 限制帧数
python traditional_method/run_traditional.py --video data/custom/videos/scene1.mp4 --frames 200

# 使用手动实现的HOG特征提取（验证算法原理，非常慢）
python traditional_method/run_traditional.py --video data/custom/videos/scene1.mp4 --no-hog-api --frames 10

# 启用GT评估
python traditional_method/run_traditional.py \
    --video data/custom/videos/scene1.mp4 \
    --eval --labels data/yolo_custom/labels/val --scene scene1
```

| 参数                  | 默认值  | 说明                            |
| ------------------- | ---- | ----------------------------- |
| `--video`           | -    | 视频文件路径                        |
| `--frames`          | 全部帧  | 最大处理帧数                        |
| `--hog-conf`        | 0.3  | HOG检测置信度阈值                    |
| `--hog-scale`       | 1.05 | HOG图像金字塔缩放系数                  |
| `--use-hog-api`     | 开启   | HOG特征提取使用OpenCV API（默认，速度快） |
| `--no-hog-api`      | -    | HOG特征提取使用手动实现（仅用于验证算法原理，非常慢） |
| `--use-svm-api`     | 开启   | SVM使用OpenCV预训练权重（默认） |
| `--no-svm-api`      | -    | SVM不使用OpenCV预训练权重（仅用于标识） |
| `--max-age`         | 15   | 轨迹最大丢失帧数                      |
| `--min-hits`        | 3    | 确认轨迹所需最小检测数                   |
| `--iou-threshold`   | 0.3  | IoU匹配阈值                       |
| `--no-optical-flow` | -    | 禁用光流运动估计                      |
| `--no-reid`         | -    | 禁用ReID重识别                     |

### 3.3.1 传统方法高级参数与速度优化

传统方法在核心算法不变的基础上，支持检测参数精细控制和速度优化：

```bash
# 速度优化：ROI裁剪（仅检测下半部，2倍加速）+ 输入缩放（75%，2-3倍加速）+ 跳帧检测
python traditional_method/run_traditional.py \
    --video data/custom/videos/scene1.mp4 \
    --roi-ratio 0.5 --input-scale 0.75 --frame-skip 2

# 自定义SVM重训练（适配特定场景）
python traditional_method/retrain_svm.py \
    --videos-dir data/custom/videos \
    --labels-base data/yolo_custom/labels/val \
    --scenes scene1 scene2
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--win-stride` | 8 | HOG滑动窗口步长，越小检测越密但越慢 |
| `--nms-threshold` | 0.45 | NMS IoU阈值，越大保留越多重叠框 |
| `--aspect-min` | 0.2 | 行人最小宽高比 |
| `--aspect-max` | 0.85 | 行人最大宽高比 |
| `--roi-ratio` | 1.0 | ROI检测比例（0.5=仅下半部，加速且减少误检） |
| `--frame-skip` | 1 | 跳帧检测（2=隔帧检测，2倍加速） |
| `--input-scale` | 1.0 | 输入缩放（0.75=缩至75%，2-3倍加速） |

### 3.4 深度方法跟踪

#### 3.4.1 使用 run_tracking_evaluation.py（推荐）

```bash
# 使用预训练模型
python run_tracking_evaluation.py --video data/custom/videos/scene1.mp4 --model pretrained

# 使用微调模型
python run_tracking_evaluation.py --video data/custom/videos/scene1.mp4 --model custom

# 限制帧数
python run_tracking_evaluation.py --video data/custom/videos/scene1.mp4 --frames 100 --model custom

# 启用GT评估
python run_tracking_evaluation.py \
    --video data/custom/videos/scene1.mp4 \
    --model custom \
    --eval --labels data/yolo_custom/labels/val --scene scene1
```

| 参数         | 默认值                             | 说明                      |
| ---------- | ------------------------------- | ----------------------- |
| `--video`  | `data/custom/videos/scene1.mp4` | 测试视频                    |
| `--frames` | 0                               | 处理帧数，0表示处理全部帧           |
| `--model`  | custom                          | 模型类型（custom/pretrained） |
| `--eval`   | -                               | 启用GT评估                  |
| `--labels` | `data/yolo_custom/labels/val`   | YOLO格式GT标注目录            |
| `--scene`  | -                               | 场景名称过滤（如scene1, scene2） |

#### 3.4.2 使用 deep_method/run_deep.py（高级）

支持单独加载 YOLO LoRA 和 ReID LoRA 权重：

```bash
# 使用预训练模型（无LoRA）
python deep_method/run_deep.py --video data/custom/videos/scene1.mp4 --frames 300

# 仅使用 YOLO LoRA 微调
python deep_method/run_deep.py --video data/custom/videos/scene3.mp4 \
    --lora runs/yolo_lora/train/weights/lora_best.pt \
    --output-dir outputs/custom/scene3 --frames 300

# 同时使用 YOLO LoRA 和 ReID LoRA 微调
python deep_method/run_deep.py --video data/custom/videos/scene3.mp4 \
    --lora runs/yolo_lora/train/weights/lora_best.pt \
    --reid-lora runs/reid/osnet_lora/best.pth \
    --output-dir outputs/custom/scene3 --frames 300

# scene4 示例
python deep_method/run_deep.py --video data/custom/videos/scene4.mp4 \
    --lora runs/yolo_lora/train/weights/lora_best.pt \
    --reid-lora runs/reid/osnet_lora/best.pth \
    --output-dir outputs/custom/scene4 --frames 300

# scene5 示例
python deep_method/run_deep.py --video data/custom/videos/scene5.mp4 \
    --lora runs/yolo_lora/train/weights/lora_best.pt \
    --reid-lora runs/reid/osnet_lora/best.pth \
    --output-dir outputs/custom/scene5 --frames 300
```

| 参数         | 说明                      |
| ---------- | ----------------------- |
| `--video`  | 视频文件路径                  |
| `--frames` | 最大处理帧数                  |
| `--lora`   | YOLO LoRA权重路径（检测器微调）   |
| `--reid-lora` | ReID LoRA权重路径（外观特征微调） |
| `--output-dir` | 输出目录                  |
| `--device` | 推理设备（cuda:0/cpu）       |
| `--conf`   | 检测置信度阈值（默认0.25）        |
| `--no-bytetrack` | 禁用ByteTrack低分框策略 |

### 3.5 对比评估（主要评估方式）

`run_comparison.py` 是项目的主要评估脚本，用于对比三种方法的性能：

- **traditional**: 传统方法 (HOG+SVM)
- **deep_custom**: 深度方法 (微调YOLO)
- **large_model**: 大模型方法 (Qwen2-VL+LoRA)

```bash
# 对所有视频运行所有方法（scene5有GT标注）
python run_comparison.py --videos-scenes scene1 scene2 scene3 scene4 scene5 --frames 150

# 只运行特定方法
python run_comparison.py --videos-scenes scene5 --methods traditional --frames 100

# MOT17数据集对比评估（默认3个序列，10帧）
python run_comparison.py --mot17 --mot17-frames 10

# 指定MOT17序列
python run_comparison.py --mot17 --mot17-seq MOT17-04-FRCNN --mot17-frames 10

# 仅对比传统和大模型方法
python run_comparison.py --mot17 --mot17-seq MOT17-02-FRCNN --methods traditional large_model

# 指定大模型推理输出目录
python run_comparison.py --mot17 --methods large_model --large-model-output outputs/large_model
```

| 参数            | 默认值          | 说明                                  |
| ------------- | ------------ | ----------------------------------- |
| `--videos-scenes` | - | 自采集视频场景名称（如scene1 scene2） |
| `--mot17`     | -            | 启用 MOT17 数据集模式                      |
| `--mot17-dir` | `data/MOT17` | MOT17 数据集根目录                        |
| `--mot17-seq` | MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-11-FRCNN | 序列名称，默认3个序列 |
| `--mot17-frames` | 10 | MOT17每个序列处理帧数 |
| `--methods`   | 全部方法 | 要运行的方法（traditional/deep_custom/large_model） |
| `--frames`    | 150 | 自采集视频处理帧数 |
| `--large-model-output` | `outputs/large_model` | 大模型方法推理输出目录 |
| `--use-hog-api` | 开启 | 传统方法HOG特征提取使用OpenCV API |
| `--no-hog-api` | - | 传统方法HOG特征提取使用手动实现 |
| `--use-svm-api` | 开启 | 传统方法SVM使用OpenCV预训练权重 |
| `--no-svm-api` | - | 传统方法SVM不使用OpenCV预训练权重 |

**评估输出：**

```
outputs/comparison/
├── comparison_report.json    # JSON格式报告
├── comparison_table.csv      # CSV格式表格
└── comparison_table.md       # Markdown格式表格
```

**评估指标**：MOTA、MOTP、IDF1、IDSW、Precision、Recall、FP、FN、TP、MT、ML、Frag、FPS

***

## 四、传统方法核心算法

### 4.1 实现方式说明

本项目传统方法支持**API模式**和**手动实现模式**两种方式：

```python
# 默认使用OpenCV API（速度快）
detector = HOGDetector(use_hog_api=True, use_svm_api=True)

# 使用手动实现（验证算法原理）
detector = HOGDetector(use_hog_api=False, use_svm_api=False)
```

**实现方式对比**：

| 模块          | HOG API模式             | HOG手动实现模式                              |
| ----------- | ---------------------- | ----------------------------------- |
| **HOG特征提取** | OpenCV `HOGDescriptor` | 手动实现Sobel梯度、Cell直方图、Block L2-Hys归一化 |
| **SVM分类器**  | 使用预训练权重，手动实现决策函数               | 使用预训练权重，手动实现决策函数                            |
| **图像金字塔**   | 手动实现                   | 手动实现                                |
| **NMS去重**   | 手动实现                   | 手动实现                                |
| **宽高比过滤**   | 手动实现                   | 手动实现                                |
| **融合框拆分**   | 手动实现                   | 手动实现                                |
| **置信度归一化**  | 手动实现                   | 手动实现                                |
| **光流计算**    | 调用API                  | 调用API                               |
| **光流校正策略**  | 手动实现                   | 手动实现                                |
| **卡尔曼滤波**   | 手动实现                   | 手动实现                                |
| **级联匹配**    | 手动实现                   | 手动实现                                |
| **代价矩阵**    | 手动实现                   | 手动实现                                |
| **匈牙利算法**   | 调用API                  | 调用API                               |
| **轨迹管理**    | 手动实现                   | 手动实现                                |

### 4.2 核心算法流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HOG + SVM 行人检测                                │
├─────────────────────────────────────────────────────────────────────┤
│  1. 图像金字塔多尺度检测 (scale=1.05)                                 │
│  2. 滑动窗口扫描 (win_stride=8x8)                                    │
│  3. HOG特征提取:                                                     │
│     - API模式: OpenCV HOGDescriptor                                  │
│     - 手动模式: Sobel梯度 + Cell直方图 + Block L2-Hys归一化           │
│  4. SVM分类（手动实现决策函数）                                       │
│  5. Sigmoid置信度归一化                                              │
│  6. 宽高比过滤 (0.2~0.85) + 融合框拆分                               │
│  7. NMS去重 (IoU=0.45)                                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    Farneback 稠密光流                                │
├─────────────────────────────────────────────────────────────────────┤
│  1. 多项式展开近似像素邻域                                           │
│  2. 最小化相邻帧像素误差估计运动向量                                  │
│  3. 输出(H,W,2)光流场，每个像素一个(dx,dy)                           │
│  4. 仅对未匹配轨迹进行光流校正（权重0.2，限幅5px）                    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    卡尔曼滤波状态预测                                │
├─────────────────────────────────────────────────────────────────────┤
│  状态向量: [cx, cy, w, h, vx, vy, vw, vh] (8维)                     │
│  观测向量: [cx, cy, w, h] (4维)                                      │
│  自适应过程噪声: 未匹配帧数越多，协方差膨胀越大                        │
│  速度衰减: vw/vh强衰减(0.1)，vx/vy温和衰减(0.7)                      │
│  宽高约束: 帧间变化≤5%，参考尺寸硬约束(0.4x~2.0x)                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    级联匹配 + 匈牙利算法                             │
├─────────────────────────────────────────────────────────────────────┤
│  1. 按time_since_update从小到大级联匹配                              │
│  2. 代价矩阵 = λ·IoU距离 + (1-λ)·马氏距离                           │
│  3. 匈牙利算法求解最优二分图匹配                                     │
│  4. 尺寸一致性惩罚：防止融合大框匹配单人轨迹                          │
│  5. 运动方向一致性惩罚                                               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    轨迹管理                                          │
├─────────────────────────────────────────────────────────────────────┤
│  创建: 未匹配检测 → 检查是否与已有轨迹重叠/交汇 → 创建TENTATIVE       │
│  确认: TENTATIVE + hits≥2 → CONFIRMED                               │
│  删除: TENTATIVE未匹配>2帧 或 CONFIRMED未匹配>15帧                   │
│  输出: 仅返回time_since_update==0的轨迹（防止幽灵框）                │
└─────────────────────────────────────────────────────────────────────┘
```

***

## 五、深度学习方法

### 5.1 YOLO11 + LoRA微调

**LoRA原理**：对冻结的卷积权重 W 添加低秩分解 W' = W + (α/r)·B·A，其中 A∈R^(r×k), B∈R^(d×r)，r=8 远小于原始维度，仅训练 A 和 B。

**训练输出：**

```
runs/yolo_lora/train/weights/
├── best.pt      # 最佳模型（标准YOLO格式，LoRA已合并，可直接推理）
├── last.pt      # 最终模型
├── lora_best.pt # LoRA权重备份（合并前）
└── lora_final.pt
```

### 5.2 多尺度IoU损失函数

```
L_box = 0.2 × L_GIoU + 0.5 × L_CIoU + 0.3 × L_EIoU
```

| IoU 变体 | 权重  | 优势                  |
| ------ | --- | ------------------- |
| GIoU   | 0.2 | 引入最小外接矩形，处理无重叠情况    |
| CIoU   | 0.5 | 考虑中心距离和宽高比，全面性最强    |
| EIoU   | 0.3 | 分离宽高损失，更高效地直接惩罚尺寸差异 |

### 5.3 OSNet ReID + LoRA微调

**联合损失**：

```
L_total = 1.0 × L_cls + 1.0 × L_tri + 0.0005 × L_center
```

| 损失函数                     | 权重     | 作用               |
| ------------------------ | ------ | ---------------- |
| CrossEntropyLabelSmooth  | 1.0    | 身份分类（ε=0.1标签平滑）  |
| TripletLoss (Batch-Hard) | 1.0    | 度量学习（margin=0.3） |
| CenterLoss               | 0.0005 | 学习类中心，增强类内紧凑性    |

***

## 六、评估指标

### 6.1 MOTA（多目标跟踪准确率）

```
MOTA = 1 - (FP + FN + IDSW) / GT
```

- FP：误检数（检测到不存在的目标）
- FN：漏检数（未检测到真实目标）
- IDSW：身份切换次数（同一目标ID变化）
- GT：真实目标总数

### 6.2 IDF1（身份F1分数）

```
IDF1 = 2 × IDTP / (2 × IDTP + IDFP + IDFN)
```

### 6.3 其他指标

- MOTP：定位精度（平均IoU）
- MT/ML：大部分跟踪成功/失败的目标数
- Frag：轨迹碎片数

***

## 七、快速开始

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

# 3. 数据转换
python tools/prepare_data.py --input data/custom --output data/yolo_custom

# 4. 模型微调（可选）
python deep_method/model/train_lora.py --data configs/dataset.yaml --epochs 30

# 5. 大模型方法（可选）
# 5a. 构建SFT数据
python model_method/build_sft_data.py --stages 1
# 5b. LoRA微调训练
python model_method/train.py --stage 1

# 6. 对比评估
python run_comparison.py --videos-scenes scene1 --frames 150
```

### 仅运行跟踪（无需训练）

```bash
# 传统方法
python traditional_method/run_traditional.py --video data/custom/videos/scene1.mp4

# 深度方法（预训练模型）
python run_tracking_evaluation.py --video data/custom/videos/scene1.mp4 --model pretrained

# 大模型方法（需先完成训练）
python model_method/visualize.py --mode detect_track --lora-path runs/stage1/final

# 对比评估
python run_comparison.py --videos-scenes scene1 --frames 150
```

***

## 八、大模型方法：Qwen2-VL 行人检测与跟踪

### 8.1 方案概述

采用**方案B: 回归原生检测格式 + 后处理跟踪**。基于 Qwen2-VL-2B-Instruct 大模型，通过 LoRA 微调实现行人检测，再通过 IoU 后处理实现跨帧跟踪。Qwen2-VL 输出原生 bbox 格式（0-1000 坐标系），后处理阶段使用 IoU 匹配关联跨帧 ID。

详细方案见 [QWEN2VL_TRACKING_PLAN.md](QWEN2VL_TRACKING_PLAN.md)。

### 8.2 模块结构

`model_method/` 目录整理后包含以下文件（按流程顺序）：

| 文件 | 类别 | 说明 |
|------|------|------|
| `build_sft_data.py` | 数据处理 | 将 MOT17 标注转换为 Qwen2-VL SFT 训练格式 (JSONL) |
| `dataset.py` | 数据处理 | SFT Dataset 类 (JSONL → Qwen2-VL 输入) |
| `lora_config.py` | 模型训练 | LoRA 配置 (r=8, alpha=16) |
| `losses.py` | 模型训练 | 自定义损失函数 (坐标加权 + IoU + 跟踪一致性) |
| `rl_reward.py` | 模型训练 | RL 奖励函数 (IoU 奖励 + ID 一致性奖励) |
| `tracking_trainer.py` | 模型训练 | 自定义 Trainer (集成辅助损失) |
| `train.py` | 模型训练 | LoRA 微调训练入口 (单帧检测SFT) |
| `visualize.py` | 模型推理 | 推理可视化+IoU评测 (detect/detect_track) |
| `evaluate.py` | 模型评估 | MOT 标准评估 (MOTA/MOTP/IDF1 等指标) |
| `ablation.py` | 模型评估 | 消融实验 (A1-A7 共 11 个配置) |

### 8.3 数据准备

将 MOT17 标注转换为 Qwen2-VL SFT 训练格式（JSONL）：

```bash
# 构建 Stage 1 单帧检测数据
python model_method/build_sft_data.py --stages 1
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mot17-root` | `data/MOT17` | MOT17 数据集根目录 |
| `--output-dir` | `data` | 输出目录 |
| `--stages` | `1` | 要构建的阶段（逗号分隔） |
| `--skip-validation` | - | 跳过数据验证 |

**生成文件**：

| 文件 | 阶段 | 说明 |
|------|------|------|
| `data/mot17_sft_stage1.jsonl` | Stage 1 | 单帧检测数据（学习 bbox 输出格式） |

**数据格式示例**（单帧检测，Qwen2-VL 0-1000 坐标系）：

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "path/to/frame.jpg"},
      {"type": "text", "text": "Detect all pedestrians in this image."}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "<ref>person</ref><box>(291,410),(371,652)</box>\n<ref>person</ref><box>(512,398),(590,640)</box>"}
    ]}
  ]
}
```

### 8.4 LoRA 微调训练

Stage 1: 单帧检测 SFT，让模型学习输出 Qwen2-VL 原生 bbox 格式（0-1000 坐标系）。

**设计理念**：采用方案B（回归原生检测格式 + 后处理跟踪），Qwen2-VL 原生支持 bbox 输出，用极小 LoRA (r=8, 仅0.4%参数) 微调单帧检测能力，跟踪通过 IoU 后处理实现。

```bash
# Stage 1: 单帧检测 SFT (LoRA r=8, 学习 bbox 输出格式)
python model_method/train.py --stage 1

# Quick 验证 (少量数据快速验证流程)
python model_method/train.py --stage 1 --quick
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage` | 1 | 训练阶段 (1=单帧检测SFT) |
| `--qlora` | 关闭 | 使用 QLoRA 4bit 量化 |
| `--no-qlora` | - | 不使用 QLoRA (fp16 训练) |
| `--epochs` | 默认 | 覆盖训练轮次 |
| `--lr` | 默认 | 覆盖学习率 |
| `--quick` | - | 快速模式 (少量数据) |

**训练输出**：

```
runs/stage1/
├── final/           # 最终 LoRA 权重
└── checkpoint-*/    # 中间检查点
```

**LoRA 配置** (极小化, 保留检测能力):

| 参数 | 值 | 说明 |
|------|-----|------|
| r | 8 | LoRA 秩 |
| alpha | 16 | 缩放因子 |
| target_modules | 7个 | q/k/v/o_proj + gate/up/down_proj |
| 可训练参数 | ~9M (0.4%) | 仅训练 LoRA 适配器 |
| 学习率 | 2e-5 | 适中, 确保学会检测格式 |

**辅助损失函数** (Stage 1 SFT):

```
L_total = L_lm + λ·L_coord_weighted
```

| 损失项 | 作用 | Stage 1 |
|--------|------|---------|
| L_lm | 标准 SFT 损失 | ✅ |
| L_coord_weighted | 坐标 token 加权 | ✅ λ=2.0 (每4步) |

### 8.5 推理可视化

在 LoRA 微调前后测试 Qwen2-VL 在 MOT17 上的检测能力，并进行 IoU 评测：

```bash
# 单帧检测模式
python model_method/visualize.py --mode detect

# 检测+IoU评测模式 (单帧检测 + IoU后处理跟踪)
python model_method/visualize.py --mode detect_track

# 指定 LoRA 权重
python model_method/visualize.py --mode detect_track --lora-path runs/stage1/final

# 指定序列和帧数
python model_method/visualize.py --mode detect_track --seq-filter MOT17-02-FRCNN --max-frames 10
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | detect | 模式 (detect/detect_track) |
| `--lora-path` | - | LoRA 权重路径 |
| `--mot17-root` | `data/MOT17` | MOT17 数据集根目录 |
| `--seq-filter` | MOT17-02-FRCNN,MOT17-11-FRCNN | 评估序列（逗号分隔） |
| `--max-frames` | 10 | 每个序列最大帧数 |
| `--output-dir` | `outputs/visualizations` | 可视化输出目录 |

### 8.6 MOT 标准评估

基于 `motmetrics` 库计算 MOTA/MOTP/IDF1/IDSW/MT/ML/FP/FN 等全套 MOT 指标。流程：加载模型 → 逐帧推理检测 → IoU 后处理跟踪 → 保存 MOT 格式 → 计算指标。

```bash
# 推理并评估指定 LoRA 权重
python model_method/evaluate.py --lora-path runs/stage1/final

# 评估原始模型 (无 LoRA)
python model_method/evaluate.py --no-lora

# 指定序列和帧数
python model_method/evaluate.py --lora-path runs/stage1/final \
    --seq-filter MOT17-02-FRCNN,MOT17-11-FRCNN --max-frames 10

# 只评估已有结果文件 (不重新推理)
python model_method/evaluate.py --eval-only --result-dir outputs/mot_results
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora-path` | `runs/stage1/final` | LoRA 权重路径 |
| `--no-lora` | - | 评估原始模型 (无 LoRA) |
| `--model-path` | `Qwen` | 基座模型路径 |
| `--mot17-root` | `data/MOT17` | MOT17 数据集根目录 |
| `--seq-filter` | MOT17-02-FRCNN,MOT17-11-FRCNN | 评估序列（逗号分隔） |
| `--max-frames` | 10 | 每个序列最大帧数 |
| `--iou-threshold` | 0.3 | IoU 后处理跟踪匹配阈值 |
| `--eval-only` | - | 只评估已有结果文件, 不重新推理 |

**输出指标**：MOTA、MOTP、IDF1、IDP、IDR、IDSW、MT、ML、FP、FN、TP、Precision、Recall、Frag

### 8.7 消融实验

实现方案文档中定义的 7 组共 11 个消融配置，自动训练 + 评估 + 生成对比报告：

```bash
# 列出所有消融配置
python model_method/ablation.py --list

# 运行所有消融实验 (快速模式, 仅 20 帧评估)
python model_method/ablation.py --quick

# 运行特定消融
python model_method/ablation.py --ablations A1,A2_r64,A4

# 跳过训练, 只评估已有权重
python model_method/ablation.py --ablations A2 --eval-only

# 只生成对比报告 (使用已有评估结果)
python model_method/ablation.py --report-only

# Dry-run (只打印命令不执行)
python model_method/ablation.py --dry-run
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--list` | - | 列出所有消融配置 |
| `--ablations` | 全部 | 要运行的消融实验 (逗号分隔) |
| `--quick` | 开启 | 快速模式 (少量帧评估) |
| `--full` | - | 完整模式 (完整训练 + 完整评估) |
| `--sequences` | 全部 | 评估序列 (逗号分隔) |
| `--train-only` | - | 只训练, 不评估 |
| `--eval-only` | - | 只评估, 不训练 |
| `--report-only` | - | 只生成对比报告 |
| `--dry-run` | - | 只打印命令不执行 |

**消融配置说明**：

| 配置 | 说明 | 需要训练 |
|------|------|----------|
| A1 | 无 LoRA (原始模型) | ❌ |
| A2_r16/r32/r64 | LoRA 秩对比 (r=16/32/64) | r16/r32 需训练 |
| A3 | 无 L_format (验证格式约束损失) | ✅ |
| A4 | 无 L_iou (验证 IoU 损失) | ❌ |
| A5 | 无 L_track (验证跟踪一致性损失) | ❌ |
| A6_stage1/2 | 训练阶段消融 (单阶段 vs 两阶段: SFT vs +RL) | ❌ |
| A7_seq2/4 | 推理窗口大小消融 (window_size=2/4) | ❌ |
| A8_format | 推理格式消融 (track vs detect_match) | ❌ |

**输出**：Markdown 对比表格 + JSON 格式报告（含 MOTA/MOTP/IDF1/IDSW 等指标对比）

***

*文档更新时间：2026年*
*项目地址：d:\学习\大创\Pedestrian_tracking*
