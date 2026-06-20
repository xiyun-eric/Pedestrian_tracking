"""
Qwen2-VL SFT Dataset 类

将 JSONL 格式的 SFT 数据加载为 PyTorch Dataset,
使用 Qwen2-VL 的 processor 进行 tokenize 和图像预处理。
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from qwen_vl_utils import process_vision_info


class TrackingSFTDataset(Dataset):
    """
    Qwen2-VL 跟踪 SFT 数据集

    从 JSONL 文件加载样本, 使用 processor 处理图像和文本。
    """

    def __init__(self, data_path, processor, max_seq_length=2048, max_pixels=None):
        """
        Args:
            data_path: JSONL 文件路径
            processor: Qwen2-VL AutoProcessor 实例
            max_seq_length: 最大序列长度 (超过则截断)
            max_pixels: 图像最大像素数 (降低分辨率以节省显存, None=使用默认)
        """
        self.data_path = Path(data_path)
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.max_pixels = max_pixels
        self.samples = self._load_data()

        # 预计算特殊 token ID, 用于高效构建 labels
        self._init_special_token_ids()

        # 如果指定了 max_pixels, 修改 processor 的图像处理器
        if max_pixels is not None:
            self.processor.image_processor.max_pixels = max_pixels
            self.processor.image_processor.min_pixels = max_pixels // 4

    def _init_special_token_ids(self):
        """预计算特殊 token ID, 避免 __getitem__ 中重复查找"""
        tokenizer = self.processor.tokenizer

        # <|im_end|> 特殊 token (用于定位 assistant 块结束位置)
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _load_data(self):
        """加载 JSONL 数据"""
        samples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                samples.append(sample)
        print(f"  加载 {len(samples)} 个样本 from {self.data_path.name}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本

        Returns:
            dict: 包含 input_ids, attention_mask, labels, pixel_values, image_grid_thw
        """
        sample = self.samples[idx]
        messages = sample['messages']

        # 使用 processor 的 chat template 格式化文本
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 处理图像
        image_inputs, video_inputs = process_vision_info(messages)

        # processor 编码
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        # 移除 batch 维度 (processor 返回 [1, ...], 需要移除第0维)
        # 注意: image_grid_thw 必须保持 2D [N, 3], 模型要求此格式
        result = {}
        for k, v in inputs.items():
            if not isinstance(v, torch.Tensor):
                result[k] = v
            elif k == 'image_grid_thw':
                # image_grid_thw 必须是 [N, 3], 不能 squeeze 成 [3]
                # 模型内部用 for t, h, w in grid_thw 迭代, 要求 2D
                if v.dim() == 1:
                    result[k] = v.unsqueeze(0)  # [3] -> [1, 3]
                else:
                    result[k] = v  # 已经是 [N, 3]
            else:
                result[k] = v.squeeze(0)

        # 构建 labels: 与 input_ids 相同, 但用户部分设为 -100
        input_ids = result['input_ids']
        labels = self._build_labels(input_ids, messages)
        result['labels'] = labels

        # 构建 id_labels: 标记每个 token 属于哪个 track ID
        # 从 assistant 回复中的 "IDx:" 模式提取 track ID
        id_labels = self._build_id_labels(input_ids, messages, labels)
        result['id_labels'] = id_labels

        # 截断到最大长度
        if result['input_ids'].size(0) > self.max_seq_length:
            for k in ['input_ids', 'attention_mask', 'labels', 'id_labels']:
                if k in result and isinstance(result[k], torch.Tensor):
                    result[k] = result[k][:self.max_seq_length]

            # 同步调整 pixel_values 和 image_grid_thw
            # 截断 input_ids 可能切掉部分 image_pad token,
            # 导致剩余 image_pad 数量与 pixel_features 数量不匹配
            result = self._truncate_image_data(result)

        return result

    def _truncate_image_data(self, result):
        """
        截断 input_ids 后, 同步调整 pixel_values 和 image_grid_thw

        原理: 统计截断后 input_ids 中剩余的 image_pad token 数量,
        保留对应数量的 pixel_features, 多余的裁掉。
        """
        image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        input_ids = result['input_ids']

        # 统计截断后剩余的 image_pad token 数
        remaining_image_tokens = (input_ids == image_pad_id).sum().item()

        if remaining_image_tokens == 0 or 'pixel_values' not in result:
            return result

        pixel_values = result['pixel_values']
        total_features = pixel_values.size(0)

        if remaining_image_tokens >= total_features:
            # 没有被截断, 不需要调整
            return result

        # 需要截断 pixel_values
        # 按 image_grid_thw 逐图像处理, 保留前面的图像
        grid_thw = result['image_grid_thw']  # [N, 3]
        kept_features = 0
        kept_images = 0

        for i in range(grid_thw.size(0)):
            t, h, w = grid_thw[i].tolist()
            n_features = t * h * w
            if kept_features + n_features <= remaining_image_tokens:
                kept_features += n_features
                kept_images += 1
            else:
                # 当前图像被部分截断, 保留能放下的部分
                partial = remaining_image_tokens - kept_features
                if partial > 0:
                    kept_features += partial
                    kept_images_partial = i + 1
                break
        else:
            kept_images_partial = kept_images

        # 截断 pixel_values
        result['pixel_values'] = pixel_values[:remaining_image_tokens]

        # 更新 image_grid_thw: 只保留完整图像的行
        # 对于部分截断的图像, 调整其 h*w 使 t*h*w = partial
        if kept_images < grid_thw.size(0):
            result['image_grid_thw'] = grid_thw[:kept_images]
            # 如果有部分截断的图像, 添加一行调整后的 grid
            if kept_features < remaining_image_tokens:
                partial = remaining_image_tokens - sum(
                    grid_thw[j, 1].item() * grid_thw[j, 2].item() * grid_thw[j, 0].item()
                    for j in range(kept_images)
                )
                if partial > 0:
                    # 简化: 将部分截断的图像作为一个新行, t=1, h*w=partial
                    import math
                    hw = partial
                    h_new = int(math.sqrt(hw))
                    w_new = hw // h_new
                    if h_new * w_new < hw:
                        w_new = hw // h_new + 1
                    new_row = torch.tensor([[1, h_new, w_new]], dtype=grid_thw.dtype)
                    result['image_grid_thw'] = torch.cat([result['image_grid_thw'], new_row], dim=0)

        return result

    def _build_labels(self, input_ids, messages):
        """
        构建 labels: 只对 assistant 回复部分计算损失

        方法: 编码不含 assistant 回复的 prompt, 得到 prompt 长度,
        将 prompt 部分设为 -100, 只保留 assistant 回复部分的 labels。

        这比 token ID 匹配更可靠, 因为不同 tokenizer 对同一文本
        的分词方式可能不同。
        """
        labels = input_ids.clone()

        # 构建 prompt-only 的 messages (去掉 assistant 回复)
        prompt_messages = [msg for msg in messages if msg['role'] != 'assistant']

        # 编码 prompt + add_generation_prompt=True
        # 这会产生: ...<|im_start|>assistant\n
        # 即 prompt 部分加上 assistant 标记
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # 用 tokenizer 编码 prompt (不含图像, 因为图像 token 不影响文本长度计算)
        prompt_ids = self.processor.tokenizer.encode(
            prompt_text, add_special_tokens=False
        )
        prompt_len = len(prompt_ids)

        # 将 prompt 部分设为 -100 (包括 <|im_start|>assistant\n 标记)
        # 这样模型只会对 assistant 的实际回复内容计算损失
        if prompt_len < len(labels):
            labels[:prompt_len] = -100
        else:
            # prompt_len >= total length, 说明有问题
            # fallback: 只保留最后 20% 的 labels
            cutoff = max(0, len(labels) - len(labels) // 5)
            labels[:cutoff] = -100

        # 同时将 <|im_end|> token 也设为 -100
        # (模型不需要学习预测结束标记)
        for i in range(len(labels)):
            if input_ids[i].item() == self.im_end_id and labels[i] != -100:
                labels[i] = -100

        return labels

    def _build_id_labels(self, input_ids, messages, labels):
        """
        构建 id_labels: 标记每个 token 属于哪个 track ID

        从 assistant 回复文本中解析 "IDx:" 模式,
        为属于该 ID 的所有 token (包括 bbox) 标注相同的 track ID。
        不属于任何 track 的 token 标记为 -100。

        这样 TrackingConsistencyLoss 就能知道哪些 token 属于同一个行人。
        """
        id_labels = torch.full_like(labels, -100)

        # 获取 assistant 回复文本
        assistant_text = ""
        for msg in messages:
            if msg['role'] == 'assistant':
                for c in msg.get('content', []):
                    if c.get('type') == 'text':
                        assistant_text += c.get('text', '')

        if not assistant_text:
            return id_labels

        # 解析文本中的 track ID 和对应位置
        # 格式: "ID2: <|object_ref_start|>person<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>"
        import re
        # 找到所有 "IDx:" 的位置
        id_pattern = r'ID(\d+):'
        id_matches = list(re.finditer(id_pattern, assistant_text))

        if not id_matches:
            # Stage 1 数据没有 track ID, 全部标记为 -100
            return id_labels

        # 为每个 ID 分配一个连续的编号 (0, 1, 2, ...)
        # 因为原始 ID 可能不连续 (如 ID2, ID3, ID8...)
        id_to_label = {}
        next_label = 0
        for m in id_matches:
            original_id = int(m.group(1))
            if original_id not in id_to_label:
                id_to_label[original_id] = next_label
                next_label += 1

        # 对 labels 中非 -100 的 token, 根据其在 assistant 回复中的位置
        # 判断属于哪个 track ID
        # 方法: 将 assistant_text 中的字符位置映射到 token 位置

        # 简化方案: 对每个 ID, 找到其对应文本片段的 token 范围
        # 通过编码 "IDx:" 前缀来定位
        tokenizer = self.processor.tokenizer

        # 对每个 track ID, 编码其对应文本片段并定位 token
        for idx, m in enumerate(id_matches):
            original_id = int(m.group(1))
            label_id = id_to_label[original_id]

            # 找到该 ID 的文本范围: 从 "IDx:" 到下一个 "IDy:" 或行尾
            start_char = m.start()
            if idx + 1 < len(id_matches):
                end_char = id_matches[idx + 1].start()
            else:
                end_char = len(assistant_text)

            id_text = assistant_text[start_char:end_char]

            # 在 input_ids 中搜索这段文本对应的 token
            # 编码该文本
            id_token_ids = tokenizer.encode(id_text, add_special_tokens=False)

            if not id_token_ids:
                continue

            # 在 input_ids 中搜索该 token 序列
            ids_list = input_ids.tolist()
            id_len = len(id_token_ids)

            for i in range(len(ids_list) - id_len + 1):
                if ids_list[i:i + id_len] == id_token_ids:
                    # 找到匹配, 标记这些 token
                    for j in range(i, i + id_len):
                        if labels[j] != -100:  # 只标记有效 token
                            id_labels[j] = label_id
                    break  # 只标记第一个匹配

        return id_labels

    def collate_fn(self, features):
        """
        自定义 collate 函数: 处理变长序列和图像

        将多个样本 pad 到同一长度, 并正确处理 pixel_values 和 image_grid_thw
        """
        # 获取各字段的 pad 值
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        # 收集所有键
        keys = set()
        for f in features:
            keys.update(f.keys())

        batch = {}
        for key in keys:
            tensors = [f[key] for f in features if key in f]
            if not tensors:
                continue

            if key == 'pixel_values':
                # pixel_values: 每个 [num_patches_i, embed_dim], 拼接为 [total_patches, embed_dim]
                batch[key] = torch.cat(tensors, dim=0)
            elif key == 'image_grid_thw':
                # image_grid_thw: 每个 [num_images_i, 3], 拼接为 [total_images, 3]
                batch[key] = torch.cat(tensors, dim=0)
            elif isinstance(tensors[0], torch.Tensor) and tensors[0].dim() >= 1:
                # 变长序列: pad 到同一长度
                max_len = max(t.size(0) for t in tensors)
                pad_value = -100 if key == 'labels' else pad_token_id
                padded = []
                for t in tensors:
                    if t.size(0) < max_len:
                        padding = torch.full(
                            (max_len - t.size(0),), pad_value,
                            dtype=t.dtype, device=t.device
                        )
                        padded.append(torch.cat([t, padding]))
                    else:
                        padded.append(t)
                batch[key] = torch.stack(padded)
            else:
                batch[key] = tensors

        return batch
