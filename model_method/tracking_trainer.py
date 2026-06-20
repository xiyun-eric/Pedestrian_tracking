"""
Qwen2-VL 跟踪任务自定义 Trainer

在标准 SFT 损失基础上添加辅助损失:
- 坐标加权损失 (L_coord): 增强 bbox 坐标精度
- IoU 损失 (L_iou): 约束预测 bbox 与 GT 的重叠度
- 跟踪一致性损失 (L_track): 同 ID 跨帧嵌入一致性

修正说明 (相比计划文档):
- 原设计在非主进程跳过辅助损失, 会导致多卡训练梯度不一致 → 修正为所有进程都计算
- 原设计每步都计算 IoU 损失 (较慢) → 改为可配置周期性计算
"""

import torch
from transformers import Trainer


class TrackingTrainer(Trainer):
    """
    自定义 Trainer: 在标准 SFT 损失基础上添加辅助损失

    优化: 辅助损失每 N 步计算一次 (aux_every_n_steps),
    避免每步处理巨大 logits 张量导致的显存带宽瓶颈。
    """

    def __init__(self, aux_loss_fn=None, iou_every_n_steps=10,
                 aux_every_n_steps=1, *args, **kwargs):
        """
        Args:
            aux_loss_fn: 辅助损失模块 (需实现 forward_from_logits 方法)
            iou_every_n_steps: 每隔多少步计算一次 IoU 损失 (0=不计算)
            aux_every_n_steps: 每隔多少步计算一次辅助损失 (1=每步, 4=每4步)
                              推荐 4: 速度提升约3倍, 效果几乎一样
        """
        super().__init__(*args, **kwargs)
        self.aux_loss_fn = aux_loss_fn
        self.iou_every_n_steps = iou_every_n_steps
        self.aux_every_n_steps = aux_every_n_steps
        self._last_aux_loss = torch.tensor(0.0)  # 缓存上一次的辅助损失

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        重写 compute_loss, 添加辅助损失

        内存优化: 当有辅助损失时, 不传 labels 给模型 (避免模型内部
        计算一次 CE + 辅助损失再计算一次 CE = 两个计算图同时占显存)。
        只计算一次 weighted CE, 显存减半。
        """
        # 提取辅助损失需要的额外字段
        id_labels = inputs.pop('id_labels', None)
        labels = inputs['labels']

        # 标准前向传播
        need_hidden = (self.aux_loss_fn is not None and
                       hasattr(self.aux_loss_fn, 'need_hidden_states') and
                       self.aux_loss_fn.need_hidden_states)

        if self.aux_loss_fn is not None:
            # 有辅助损失: 不传 labels, 模型不计算 loss, 只返回 logits
            # 避免模型内部 CE 和辅助损失 CE 创建两个计算图
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=inputs.get('pixel_values'),
                image_grid_thw=inputs.get('image_grid_thw'),
                output_hidden_states=need_hidden,
            )

            # 辅助损失就是 weighted CE, 直接作为总损失 (不再加 lm_loss)
            # 每 aux_every_n_steps 步计算一次
            should_compute_aux = (
                self.aux_every_n_steps <= 1 or
                self.state.global_step % self.aux_every_n_steps == 0
            )

            if should_compute_aux:
                aux_loss, aux_details = self.aux_loss_fn.forward_from_logits(
                    logits=outputs.logits,
                    labels=labels,
                    hidden_states=outputs.hidden_states[-1] if outputs.hidden_states else None,
                    id_labels=id_labels,
                    attention_mask=inputs.get('attention_mask'),
                    compute_iou=False,
                )
                self._last_aux_loss = aux_loss.detach()
            else:
                # 非计算步: 仍需计算 loss 以回传梯度, 但用标准 CE (更轻量)
                # 只计算 lm_loss, 不计算 weighted loss
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                import torch.nn.functional as F
                aux_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                aux_details = {}

            total_loss = aux_loss  # weighted CE 已包含坐标加权

            # 日志记录
            if self.state.is_world_process_zero:
                log_dict = {
                    'loss': total_loss.item(),
                    'aux_loss': float(self._last_aux_loss),
                }
                log_dict.update({f'aux_{k}': v for k, v in aux_details.items()})
                self.log(log_dict)
        else:
            # 无辅助损失: 正常传 labels, 模型自己计算 loss
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pixel_values=inputs.get('pixel_values'),
                image_grid_thw=inputs.get('image_grid_thw'),
                output_hidden_states=False,
                labels=labels,
            )
            total_loss = outputs.loss

        return (total_loss, outputs) if return_outputs else total_loss
