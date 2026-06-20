"""
OSNet ReID 模型的 LoRA (Low-Rank Adaptation) 微调

OSNet 网络结构:
  - 由多个 OSBlock 组成，每个 OSBlock 包含多个 Stream
  - Stream 内部使用 Conv2d(1x1) 做通道降维/升维
  - 最后通过 classifier (Linear) 做身份分类
  - 特征提取时 forward 返回 512 维向量（跳过 classifier）

LoRA 适配策略:
  1. 对 OSBlock 中的 1x1 Conv2d 注入 LoRA（与 YOLO 的 Conv2dLoRA 类似）
  2. 对 classifier Linear 层注入 LoRA（标准 LoRA）
  3. 只微调深层（layer3, layer4），保持浅层特征不变

适用场景:
  - r=4  ~ 少量参数，适合小数据集
  - r=8  ~ 平衡精度与参数量
  - r=16 ~ 更强表达能力，适合大数据集
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict


class LinearLoRA(nn.Module):
    """
    Linear 层的 LoRA 适配

    output = original_linear(x) + (alpha/r) * lora_B(lora_A(x))

    其中:
    - lora_A: 降维 (in_features -> r)
    - lora_B: 升维 (r -> out_features)
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = linear.in_features
        out_features = linear.out_features

        # 冻结原始权重
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化: A 用 Kaiming，B 用零（保证初始输出不变）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.linear(x)
        lora_out = self.dropout(x) @ self.lora_A.T  # (B, r)
        lora_out = lora_out @ self.lora_B.T           # (B, out_features)
        return orig_out + lora_out * self.scaling

    def merge_weights(self):
        """将 LoRA 权重合并到原始 Linear 层"""
        merged_weight = self.linear.weight.data + (self.lora_B @ self.lora_A) * self.scaling
        self.linear.weight.data = merged_weight


class Conv2dLoRA(nn.Module):
    """
    Conv2d(1x1) 的 LoRA 适配（OSNet 专用）

    OSBlock 中的 1x1 卷积用于通道变换，等价于逐像素 Linear，
    适配标准 LoRA 低秩分解。
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert conv.kernel_size == (1, 1), \
            f"Conv2dLoRA 仅支持 kernel_size=1，当前为 {conv.kernel_size}"

        self.conv = conv
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_channels = conv.in_channels
        out_channels = conv.out_channels
        groups = conv.groups

        # 冻结原始权重
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False

        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(r, in_channels // groups))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.conv(x)

        # LoRA 分支: 1x1 conv 等价于矩阵乘法
        lora_out = F.conv2d(
            x,
            self.lora_A.unsqueeze(-1).unsqueeze(-1),
            groups=self.conv.groups,
        )
        lora_out = self.dropout(lora_out)
        lora_out = F.conv2d(
            lora_out,
            self.lora_B.unsqueeze(-1).unsqueeze(-1),
        )

        return orig_out + lora_out * self.scaling

    def merge_weights(self):
        """将 LoRA 权重合并到原始 Conv2d"""
        merged_weight = self.conv.weight.data + \
            (self.lora_B @ self.lora_A).view_as(self.conv.weight) * self.scaling
        self.conv.weight.data = merged_weight


def inject_lora_to_osnet(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    target_layers: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, int]:
    """
    向 OSNet 模型注入 LoRA 层

    策略:
    1. 对 OSBlock 中的 1x1 Conv2d 注入 Conv2dLoRA
    2. 对 classifier Linear 层注入 LinearLoRA
    3. 默认只微调深层 (layer3, layer4, classifier)

    Args:
        model: torchreid OSNet 模型
        r: LoRA 秩
        alpha: LoRA 缩放系数
        dropout: dropout 率
        target_layers: 目标层名称列表（None=自动选择深层）
        verbose: 是否打印注入信息

    Returns:
        (模型, 可训参数数量)
    """
    if target_layers is None:
        # 默认只微调深层特征提取，不修改 classifier
        # OSNet 的模块命名是 conv1, conv2, conv3, conv4, conv5, fc, classifier
        # 不注入 classifier 的原因：
        #   1. classifier 形状 4101→59 会导致过拟合
        #   2. 推理时 classifier 被跳过，只用到 512 维特征
        #   3. 保持预训练分类能力，通过 LoRA 微调特征提取层即可
        target_layers = ['conv3', 'conv4']

    injected_count = 0
    trainable_params = 0

    for name, module in model.named_modules():
        # 处理 Linear 层（classifier）
        if isinstance(module, nn.Linear):
            should_inject = any(t in name for t in target_layers)
            if not should_inject:
                continue

            name_parts = name.split('.')
            parent = model
            for part in name_parts[:-1]:
                parent = getattr(parent, part)
            child_name = name_parts[-1]

            lora_layer = LinearLoRA(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, lora_layer)

            trainable_params += r * (module.in_features + module.out_features)
            injected_count += 1

            if verbose:
                print(f"  [LoRA] 注入 Linear: {name} (in={module.in_features}, out={module.out_features})")

        # 处理 1x1 Conv2d 层（OSBlock 内部）
        elif isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1):
            should_inject = any(t in name for t in target_layers)
            if not should_inject:
                continue

            name_parts = name.split('.')
            parent = model
            for part in name_parts[:-1]:
                parent = getattr(parent, part)
            child_name = name_parts[-1]

            lora_layer = Conv2dLoRA(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, lora_layer)

            trainable_params += r * (module.in_channels + module.out_channels)
            injected_count += 1

            if verbose:
                print(f"  [LoRA] 注入 Conv2d: {name} (in={module.in_channels}, out={module.out_channels})")

    if verbose:
        total = sum(p.numel() for p in model.parameters())
        print(f"\n  总计: 注入 {injected_count} 个 LoRA 层")
        print(f"  可训参数: {trainable_params:,} / {total:,} ({100*trainable_params/total:.2f}%)")

    return model, trainable_params


def freeze_model_except_lora(model: nn.Module, verbose: bool = True):
    """
    冻结所有非 LoRA 参数，只训练 LoRA 的 lora_A 和 lora_B
    """
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[LoRA 参数统计] 可训: {trainable:,} / 总计: {total:,} ({100*trainable/total:.2f}%)")


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """获取所有 LoRA 权重"""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state[name] = param.data.clone()
    return lora_state


def save_lora_weights(model: nn.Module, path: str):
    """保存 LoRA 权重（轻量级）"""
    lora_state = get_lora_state_dict(model)
    torch.save(lora_state, path)
    print(f"[LoRA] 权重已保存到: {path} ({len(lora_state)} 个参数组)")


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """加载 LoRA 权重到已注入 LoRA 的模型"""
    lora_state = torch.load(path, map_location='cpu')
    model_state = model.state_dict()

    loaded = 0
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
            loaded += 1

    model.load_state_dict(model_state, strict=False)
    print(f"[LoRA] 已加载 {loaded}/{len(lora_state)} 个 LoRA 参数组: {path}")
    return model


def merge_lora_to_model(model: nn.Module) -> nn.Module:
    """
    将所有 LoRA 权重合并回原始层，替换为标准层

    合并后模型不再包含 LoRA 层，可用 torchreid 标准方式加载。
    """
    modules_to_replace = []

    for name, module in model.named_modules():
        if isinstance(module, LinearLoRA):
            module.merge_weights()
            modules_to_replace.append((name, module.linear))
        elif isinstance(module, Conv2dLoRA):
            module.merge_weights()
            modules_to_replace.append((name, module.conv))

    # 替换 LoRA 层为原始层
    for name, original_layer in modules_to_replace:
        name_parts = name.split('.')
        parent = model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        child_name = name_parts[-1]
        setattr(parent, child_name, original_layer)

    print(f"[LoRA] 已合并 {len(modules_to_replace)} 个 LoRA 层到原始层")
    return model


def save_merged_model(model: nn.Module, path: str, num_classes: int = 1000):
    """
    合并 LoRA 权重后保存完整的 OSNet state_dict

    保存格式与 torchreid 预训练权重一致，可直接用 ReIDExtractor 加载。
    """
    model = merge_lora_to_model(model)
    state_dict = model.state_dict()
    torch.save(state_dict, path)
    print(f"[LoRA] 合并后完整模型已保存到: {path}")
