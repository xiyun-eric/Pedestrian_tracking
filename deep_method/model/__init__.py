"""
Deep Learning Model Package

模块:
  - lora_layers: YOLO LoRA 适配层
  - custom_loss: YOLO 自定义损失函数
  - train_lora: YOLO LoRA 微调训练
  - train_yolo: YOLO 标准训练
  - osnet_lora: OSNet ReID LoRA 适配层
  - reid_loss: ReID 专用损失函数
  - train_osnet_lora: OSNet LoRA 微调训练
"""

# YOLO 相关
from deep_method.model.lora_layers import (
    Conv2dLoRA,
    inject_lora_to_yolo,
    freeze_model_except_lora as freeze_yolo_except_lora,
    save_lora_weights as save_yolo_lora_weights,
    load_lora_weights as load_yolo_lora_weights,
    merge_lora_to_model as merge_yolo_lora,
)

from deep_method.model.custom_loss import (
    MultiScaleIoULoss,
    FocalLoss,
    MotionConsistencyLoss,
    DepthAwareLoss,
    AdaptiveWeightLoss,
)

# OSNet ReID 相关
from deep_method.model.osnet_lora import (
    LinearLoRA,
    Conv2dLoRA as OSNetConv2dLoRA,
    inject_lora_to_osnet,
    freeze_model_except_lora as freeze_osnet_except_lora,
    save_lora_weights as save_osnet_lora_weights,
    load_lora_weights as load_osnet_lora_weights,
    merge_lora_to_model as merge_osnet_lora,
    save_merged_model as save_osnet_merged_model,
)

from deep_method.model.reid_loss import (
    TripletLoss,
    CenterLoss,
    CrossEntropyLabelSmooth,
    CombinedReIDLoss,
)

__all__ = [
    # YOLO LoRA
    'Conv2dLoRA',
    'inject_lora_to_yolo',
    'freeze_yolo_except_lora',
    'save_yolo_lora_weights',
    'load_yolo_lora_weights',
    'merge_yolo_lora',

    # YOLO 自定义损失
    'MultiScaleIoULoss',
    'FocalLoss',
    'MotionConsistencyLoss',
    'DepthAwareLoss',
    'AdaptiveWeightLoss',

    # OSNet LoRA
    'LinearLoRA',
    'OSNetConv2dLoRA',
    'inject_lora_to_osnet',
    'freeze_osnet_except_lora',
    'save_osnet_lora_weights',
    'load_osnet_lora_weights',
    'merge_osnet_lora',
    'save_osnet_merged_model',

    # ReID 损失
    'TripletLoss',
    'CenterLoss',
    'CrossEntropyLabelSmooth',
    'CombinedReIDLoss',
]
