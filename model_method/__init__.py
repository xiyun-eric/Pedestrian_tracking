"""
model_method: Qwen2-VL 端到端行人多目标跟踪

模块组成:
  - verify_model.py       模型验证 (加载与单帧检测)
  - build_sft_data.py     SFT 数据构建 (MOT17 -> JSONL)
  - train.py              LoRA 微调训练 (两阶段: 混合数据SFT + GRPO RL)
  - visualize.py          推理可视化 (detect/track/track10/track_dm/compare)
  - evaluate.py           MOT 标准评估 (MOTA/MOTP/IDF1, 支持track/detect_match格式)
  - ablation.py           消融实验 (A1-A8)
  - dataset.py            SFT Dataset (JSONL -> Qwen2-VL 输入)
  - lora_config.py        LoRA 配置
  - losses.py             自定义损失函数
  - rl_reward.py          RL 奖励函数
  - tracking_trainer.py   自定义 Trainer
"""
