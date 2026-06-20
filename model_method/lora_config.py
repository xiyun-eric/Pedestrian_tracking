"""
Qwen2-VL 跟踪任务 LoRA 配置

优化策略:
- r=8 (极小LoRA), 只修改0.4%参数, 保留基座模型检测能力
- alpha=16, alpha/r=2, 适度缩放
- 7个target模块覆盖注意力和FFN层
"""

from peft import LoraConfig, TaskType


def get_tracking_lora_config(r=8, alpha=16, dropout=0.05):
    """
    Qwen2-VL 跟踪任务 LoRA 配置

    Args:
        r: LoRA 秩, r=8 极小, 保留基座模型检测能力
        alpha: 缩放因子, alpha/r=2
        dropout: Dropout 比率
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            # 注意力层 (核心)
            "q_proj", "k_proj", "v_proj", "o_proj",
            # FFN 层 (辅助, 增强空间理解)
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    return config


def get_tracking_lora_config_small():
    """超轻量 LoRA 配置 (r=4), 最大程度保留基座能力"""
    return get_tracking_lora_config(r=4, alpha=8, dropout=0.05)


def get_tracking_lora_config_medium():
    """中等 LoRA 配置 (r=16), 平衡性能与保留"""
    return get_tracking_lora_config(r=16, alpha=32, dropout=0.05)


# 各阶段推荐配置
STAGE_LORA_CONFIGS = {
    1: get_tracking_lora_config,       # Stage 1: r=8, SFT 跟踪
    2: get_tracking_lora_config,       # Stage 2: r=8, GRPO RL
}
