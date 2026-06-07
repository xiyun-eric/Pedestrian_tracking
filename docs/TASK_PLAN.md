# 项目任务规划

本文档记录了行人跟踪项目的六阶段任务规划及各阶段详细内容。

***

## Phase 1：数据整理 ✅ 已完成

### 任务目标

整理自采集数据集，转换为 YOLO 格式，创建配置文件。

### 任务详情

| 任务ID | 任务内容                 | 说明                                             | 状态                          |
| ---- | -------------------- | ---------------------------------------------- | --------------------------- |
| 1.1  | 检查并修复数据目录结构          | 确保 images/scene1/val 对应 annotations/scene1/val | ✅ 完成                        |
| 1.2  | 统计标注数据质量             | 检查标注框数量、类别分布、是否有空标注                            | ✅ 完成                        |
| 1.3  | 创建 dataset.yaml 配置文件 | 用于 YOLO 训练的数据集配置                               | ✅ 完成                        |
| 1.4  | 划分 train/val/test    | 确保图像和标注对应，比例建议 8:1:1                           | ✅ 完成 (train:2573, val:1268) |

### 目录结构要求

```
data/yolo_custom/
├── images/
│   ├── train/     # 训练集图像
│   └── val/       # 验证集图像
├── labels/
│   ├── train/     # 训练集标签（YOLO格式）
│   └── val/       # 验证集标签（YOLO格式）
└── dataset.yaml   # 数据集配置文件
```

### YOLO 标签格式

每行格式：`class_id center_x center_y width height`

```
0 0.512345 0.543210 0.098765 0.234567
```

| 字段        | 含义         | 范围                      |
| --------- | ---------- | ----------------------- |
| class\_id | 类别 ID      | 0=行人, 1=骑车人, 2=车辆, 3=其他 |
| center\_x | 边界框中心 x 坐标 | 0.0 \~ 1.0 (归一化)        |
| center\_y | 边界框中心 y 坐标 | 0.0 \~ 1.0 (归一化)        |
| width     | 边界框宽度      | 0.0 \~ 1.0 (归一化)        |
| height    | 边界框高度      | 0.0 \~ 1.0 (归一化)        |

***

## Phase 2：LoRA 微调训练 ✅ 已完成

### 任务目标

使用自采集数据集对 YOLO11 模型进行微调训练。

### 任务详情

| 任务ID | 任务内容       | 说明                                           |
| ---- | ---------- | -------------------------------------------- |
| 2.1  | 配置训练参数     | batch\_size、epochs、learning\_rate、lora\_rank |
| 2.2  | 运行 LoRA 微调 | 使用自采集数据集训练                                   |
| 2.3  | 保存 LoRA 权重 | 输出到 weights/yolo\_custom/ 目录                 |
| 2.4  | 验证微调效果     | 对比原始模型 vs 微调模型的检测精度                          |

### 训练参数建议

| 参数          | 推荐值   | 说明           |
| ----------- | ----- | ------------ |
| epochs      | 30-50 | 训练轮数         |
| batch\_size | 4-8   | 批次大小（根据显存调整） |
| img\_size   | 640   | 输入图像尺寸       |
| lr0         | 0.001 | 初始学习率        |
| device      | 0     | GPU 设备编号     |

### 训练命令

```bash
python deep_method/model/train_yolo.py \
    --data data/yolo_custom/dataset.yaml \
    --epochs 30 \
    --batch 4 \
    --device 0
```

### 输出文件

训练完成后，模型保存在：

- `weights/yolo_custom/train/best.pt` - 最佳模型
- `weights/yolo_custom/train/last.pt` - 最终模型

***

## Phase 3：跟踪算法改进 ✅ 已完成

### 任务目标

完善深度学习跟踪管道，提升行人跟踪精度和鲁棒性。

### 最终方案：AdvancedTracker + standard预设

采用 **AdvancedTracker** 整合所有改进技术，使用 **standard预设** 作为通用配置，无需针对不同场景频繁调整参数。

### 整合的改进技术

| 改进 | 效果 |
| --- | --- |
| ByteTrack二次匹配 | 低分框二次匹配，减少遮挡导致的轨迹断裂 |
| ReID外观特征（OSNet x1.0 + MSMT17预训练） | 行人重识别专用特征，减少ID切换 |
| 融合代价矩阵 | IoU(30%) + 马氏距离(20%) + 外观(50%) |
| 动态外观权重 | 空间重叠时降低外观权重，解决侧背身ID切换 |
| 社会行为约束 | 防止目标重叠，减少密集场景ID切换 |
| 尺度一致性约束 | 防止近/远处目标ID混淆，防止姿态变化产生重复ID |
| 运动一致性约束 | 防止静止目标被运动目标干扰导致ID切换 |
| ReID重匹配（匈牙利算法） | 遮挡恢复时全局最优匹配，防止ID混淆 |
| 新轨迹重复检查 | 位置+尺度重叠时不创建新轨迹，防止同一目标多ID |

### 推荐配置

| 参数 | 推荐值 | 说明 |
| --- | --- | --- |
| 跟踪器 | AdvancedTracker | 集成所有改进的增强版跟踪器 |
| 预设 | standard | 通用平衡配置 |
| ReID模型 | OSNet x1.0 + MSMT17预训练 | 行人重识别专用，权重文件 `weights/osnet_x1_0_msmt17.pth` |
| 检测器 | YOLO11m / 本地微调模型 | 行人检测 |

### 使用示例

```python
from deep_method.tracking import create_tracker
from deep_method.detector import YOLODetector

# 创建检测器
detector = YOLODetector(
    model_path='yolo11m.pt',
    device='cpu',
    conf_threshold=0.1,
    iou_threshold=0.5,
    classes=[0],  # 行人
)

# 创建跟踪器（standard预设，启用ReID）
tracker = create_tracker(
    tracker_type='advanced',
    preset='standard',
    use_reid=True,
    reid_model='osnet_x1_0',
    device='cpu',
    use_torchreid=True,
)

# 跟踪流程
for frame in video_frames:
    detections, confidences, _ = detector.detect(frame)
    # ByteTrack: 分离高/低分框
    high_mask = confidences >= 0.5
    high_dets, high_confs = detections[high_mask], confidences[high_mask]
    low_dets, low_confs = detections[~high_mask], confidences[~high_mask]
    # ReID特征提取（仅高分框）
    features = tracker.reid_extractor.extract_features_batch(frame, high_dets)
    # 跟踪更新
    tracks = tracker.update(
        high_dets, high_confs,
        features=features, image=frame,
        low_conf_detections=low_dets,
        low_conf_confidences=low_confs,
    )
    for track in tracks:
        if track.is_confirmed and track.time_since_update == 0:
            bbox = track.get_bbox()
            tid = track.track_id
```

### 实现文件

| 文件 | 说明 |
| --- | --- |
| `deep_method/tracking/advanced_tracker.py` | 增强版跟踪器，整合所有改进 |
| `deep_method/tracking/reid_extractor.py` | ReID外观特征提取（OSNet/torchreid） |
| `deep_method/tracking/tracking_config.py` | 参数配置与预设管理 |
| `deep_method/tracking/__init__.py` | 跟踪模块入口与create_tracker工厂 |
| `weights/osnet_x1_0_msmt17.pth` | OSNet x1.0 MSMT17预训练权重 |
| `weights/osnet_x0_25_msmt17.pt` | OSNet x0.25 MSMT17预训练权重（轻量备选） |

***

## Phase 4：评估流程建立 ✅ 已完成

### 任务目标

建立完整的评估流程，自动计算跟踪指标。

### 任务详情

| 任务ID | 任务内容 | 说明 | 状态 |
| --- | --- | --- | --- |
| 4.1 | YOLO GT转MOT格式 | 将YOLO归一化标注转换为像素坐标MOT格式 | ✅ 完成 |
| 4.2 | 标准评估指标计算 | MOTA/MOTP/IDF1/IDSW/MT/ML/Frag/FP/FN/TP | ✅ 完成 |
| 4.3 | 评估报告输出 | JSON/CSV格式报告 + MOT格式预测/GT文件 | ✅ 完成 |
| 4.4 | 集成评估到跟踪管道 | run_tracking_evaluation.py --eval 一键评估 | ✅ 完成 |

### 评估指标说明

| 指标 | 含义 | 计算方式 |
| --- | --- | --- |
| MOTA | 多目标跟踪准确度 | 1 - (FN + FP + IDSW) / GT |
| MOTP | 多目标跟踪精度 | 平均IoU |
| IDF1 | 身份保持率 | 2*IDTP / (2*IDTP + IDFP + IDFN) |
| IDSW | ID切换次数 | GT目标匹配的pred ID发生变化 |
| MT | 大部分跟踪成功 | GT目标被跟踪>80%帧数 |
| ML | 大部分跟踪失败 | GT目标被跟踪<20%帧数 |
| Frag | 轨迹碎片数 | GT目标从匹配变为不匹配的次数 |
| FP | 误检数 | 预测框无GT匹配 |
| FN | 漏检数 | GT框无预测匹配 |

### 使用方式

```bash
# 方式1：测试脚本 + 评估（推荐）
python test_tracking_phase3.py \
    --video data/custom/videos/scene2.mp4 \
    --frames 100 \
    --eval \
    --labels data/yolo_custom/labels/val \
    --scene scene2

# 方式2：独立评估命令
python tools/evaluate.py \
    --video data/custom/videos/scene2.mp4 \
    --labels data/yolo_custom/labels/val \
    --scene scene2 \
    --frames 100 \
    --output outputs/evaluation

# 方式3：仅跟踪不评估
python test_tracking_phase3.py \
    --video data/custom/videos/scene2.mp4 \
    --frames 100
```

### 评估输出

```
outputs/evaluation/
├── evaluation_report.json    # 评估指标（JSON）
├── evaluation_report.csv     # 评估指标（CSV）
├── predictions.txt           # 预测结果（MOT格式）
└── gt.txt                    # GT标注（MOT格式）
```

### 评估报告示例

```json
{
  "MOTA": 0.75,
  "MOTP": 0.82,
  "IDF1": 0.68,
  "IDSW": 15,
  "FP": 120,
  "FN": 80,
  "TP": 800,
  "GT": 1000,
  "Precision": 0.87,
  "Recall": 0.91,
  "MT": 5,
  "ML": 2,
  "Frag": 20,
  "num_frames": 100,
  "num_gt_ids": 8,
  "num_pred_ids": 12
}
```

### 实现文件

| 文件 | 说明 |
| --- | --- |
| `tools/evaluate.py` | 评估核心模块（TrackingEvaluator + TrackingMetrics + evaluate_tracking） |
| `test_tracking_phase3.py` | 测试脚本，支持 `--eval` 参数集成评估 |

### 注意事项

- YOLO标注没有全局目标ID，同一目标在不同帧中会被分配不同ID，因此IDF1/IDSW主要反映检测精度而非ID一致性
- 如需精确的ID一致性评估，需要使用带全局ID的GT标注（如KITTI Tracking格式）

***

## Phase 5：对比实验

### 任务目标

对比传统方法与深度学习方法的跟踪效果。

### 任务详情

| 任务ID | 任务内容       | 说明                       | 入口脚本 |
| ---- | ---------- | ------------------------ | -------- |
| 5.1  | 运行传统方法     | HOG+SVM + 光流 + 卡尔曼 + 匈牙利 | `traditional_method/run_traditional.py --video` |
| 5.2  | 运行深度方法（原始） | YOLO11m 预训练 + AdvancedTracker | `run_tracking_evaluation.py --model pretrained` |
| 5.3  | 运行深度方法（微调） | YOLO11m 微调 + AdvancedTracker | `run_tracking_evaluation.py --model custom` |
| 5.4  | 对比分析       | MOTA/MOTP/FP/FN/FPS 对比表格 + 可视化 | `run_comparison.py`（需新建） |

### 实验对比表格模板

| 方法             | MOTA | MOTP | FP | FN | IDSW* | FPS |
| -------------- | ---- | ---- | -- | -- | ----- | --- |
| 传统方法 (HOG+SVM) | -    | -    | -  | -  | -     | -   |
| 深度方法 (原始)      | -    | -    | -  | -  | -     | -   |
| 深度方法 (微调)      | -    | -    | -  | -  | -     | -   |

> *IDSW 为参考指标，受 GT 标注局限性影响

### 运行命令

```bash
# 传统方法
python traditional_method/run_traditional.py \
    --video data/custom/videos/scene1.mp4 \
    --frames 200

# 深度方法（原始）
python run_tracking_evaluation.py \
    --video data/custom/videos/scene1.mp4 \
    --model pretrained \
    --eval --labels data/yolo_custom/labels/val --scene scene1

# 深度方法（微调）
python run_tracking_evaluation.py \
    --video data/custom/videos/scene1.mp4 \
    --model custom \
    --eval --labels data/yolo_custom/labels/val --scene scene1

# 一键对比实验（需新建 run_comparison.py）
python run_comparison.py \
    --videos data/custom/videos/scene1.mp4 \
    --labels data/yolo_custom/labels/val \
    --methods all \
    --frames 200
```

### 需新建文件

| 文件 | 说明 |
| --- | --- |
| `run_comparison.py` | 实验主控脚本，统一调度三种方法 + 生成对比报告 |
| `tools/comparison.py` | 对比分析模块，生成表格/图表/报告 |

### 需修改文件

| 文件 | 修改内容 |
| --- | --- |
| `traditional_method/tracking_pipeline.py` | `process_video()` 增加评估数据收集逻辑 |
| `run_tracking_evaluation.py` | `run_tracking()` 增加返回值（Dict） |

