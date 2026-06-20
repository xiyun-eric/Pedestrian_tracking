"""
YOLOv11 行人检测器的 LoRA (Low-Rank Adaptation) 微调

LoRA 原理:
  对于预训练权重矩阵 W ∈ R^{d×k}，冻结 W，添加低秩分解:
    W' = W + B·A  (其中 B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k))
  
  训练时只更新 A 和 B，大幅减少可训参数（通常减少99%以上）

YOLO 中的 LoRA 适配:
  - YOLOv11 大量使用 Conv2d(kernel=1) 作为特征融合
  - Conv2d(kernel=1) 等价于全连接层，适配标准 LoRA
  - 对 C2f 模块中的 1x1 卷积层注入 LoRA
  - 只微调检测头附近的层，保持 backbone 特征不变

适用场景 (8GB VRAM):
  - r=4  ~  0.2M 可训参数
  - r=8  ~  0.4M 可训参数
  - r=16 ~  0.8M 可训参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple


class Conv2dLoRA(nn.Module):
    """
    Conv2d 的 LoRA 适配层
    
    对 kernel_size=1 的 Conv2d 应用低秩分解:
      output = original_conv(x) + (alpha/r) * lora_B(lora_A(x))
    
    其中:
    - lora_A: 降维层 (in_channels -> r)
    - lora_B: 升维层 (r -> out_channels)
    - r: 秩 (rank)，通常为 4, 8, 16
    - alpha: 缩放系数
    """
    
    def __init__(
        self,
        conv: nn.Conv2d,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            conv: 原始卷积层 (kernel_size 必须为 1)
            r: LoRA 秩
            alpha: 缩放系数
            dropout: dropout 率
        """
        super().__init__()
        
        assert conv.kernel_size == (1, 1), \
            f"Conv2dLoRA 仅支持 kernel_size=1 的卷积层，当前为 {conv.kernel_size}"
        
        self.conv = conv  # 冻结的原始层
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
        # lora_A: 降维，输入通道 -> r
        self.lora_A = nn.Parameter(torch.zeros(r, in_channels // groups))
        # lora_B: 升维，r -> 输出通道
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化: Kaiming 均匀分布
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始卷积输出
        orig_out = self.conv(x)
        
        # LoRA 分支
        # lora_A: (in_c // groups) -> r
        # x shape: (B, C_in, 1, 1) for 1x1 conv
        lora_out = F.conv2d(
            x,
            self.lora_A.unsqueeze(-1).unsqueeze(-1),  # (r, in_c // g, 1, 1)
            groups=self.conv.groups,
        )
        lora_out = self.dropout(lora_out)
        
        # lora_B: r -> out_c
        lora_out = F.conv2d(
            lora_out,
            self.lora_B.unsqueeze(-1).unsqueeze(-1),  # (out_c, r, 1, 1)
        )
        
        return orig_out + lora_out * self.scaling
    
    # 属性代理：让 Conv2dLoRA 兼容 YOLO 的 fuse 操作
    @property
    def weight(self):
        """代理到内部 conv 的 weight"""
        return self.conv.weight
    
    @property
    def bias(self):
        """代理到内部 conv 的 bias"""
        return self.conv.bias
    
    @property
    def out_channels(self):
        return self.conv.out_channels
    
    @property
    def in_channels(self):
        return self.conv.in_channels
    
    @property
    def kernel_size(self):
        return self.conv.kernel_size
    
    @property
    def stride(self):
        return self.conv.stride
    
    @property
    def padding(self):
        return self.conv.padding
    
    @property
    def groups(self):
        return self.conv.groups
    
    @classmethod
    def from_conv(cls, conv: nn.Conv2d, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        """从现有 Conv2d 创建 LoRA 版本"""
        return cls(conv, r=r, alpha=alpha, dropout=dropout)
    
    def merge_weights(self):
        """将 LoRA 权重合并到原始卷积中（用于推理加速）"""
        if self.conv.weight.requires_grad:
            return  # 已经合并
        
        # 计算合并后的权重
        merged_weight = self.conv.weight.data + \
            (self.lora_B @ self.lora_A).view_as(self.conv.weight) * self.scaling
        
        # 更新原始权重
        self.conv.weight.data = merged_weight
        self.conv.weight.requires_grad = False


def inject_lora_to_yolo(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[nn.Module, int]:
    """
    向 YOLOv11 模型注入 LoRA 层
    
    策略:
    1. 找到所有 kernel_size=1 的 Conv2d 层
    2. 只对检测头附近的层注入 LoRA（避免影响 backbone）
    3. 可选指定目标模块名称模式
    
    Args:
        model: YOLO 模型
        r: LoRA 秩
        alpha: LoRA 缩放系数
        target_modules: 目标模块名称列表 (None=自动选择检测头)
        verbose: 是否打印注入信息
    
    Returns:
        (模型, 可训参数数量)
    """
    if target_modules is None:
        target_modules = ['detect', 'cv4', 'cv3', 'cv2']
    
    injected_count = 0
    trainable_params = 0
    
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if module.kernel_size != (1, 1):
            continue
        
        # 检查是否在目标模块中
        should_inject = any(t in name for t in target_modules)
        if not should_inject:
            continue
        
        # 获取父模块
        name_parts = name.split('.')
        parent = model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        
        child_name = name_parts[-1]
        
        # 创建 LoRA 层并替换
        lora_conv = Conv2dLoRA(module, r=r, alpha=alpha)
        setattr(parent, child_name, lora_conv)
        
        trainable_params += r * (module.in_channels + module.out_channels)
        injected_count += 1
        
        if verbose:
            print(f"  [LoRA] 注入: {name} (in={module.in_channels}, out={module.out_channels})")
    
    if verbose:
        total = sum(p.numel() for p in model.parameters())
        print(f"\n  总计: 注入 {injected_count} 个 LoRA 层")
        print(f"  可训参数: {trainable_params:,} / {total:,} ({100*trainable_params/total:.2f}%)")
    
    return model, trainable_params


def freeze_model_except_lora(model: nn.Module):
    """
    冻结所有非 LoRA 参数
    
    只训练 LoRA 的 lora_A 和 lora_B 参数，
    其他所有参数（包括BN、原始卷积等）全部冻结。
    """
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # 打印可训参数统计
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA 参数统计] 可训: {trainable:,} / 总计: {total:,} ({100*trainable/total:.2f}%)")


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """获取所有 LoRA 参数"""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
    return lora_params


def save_lora_weights(model: nn.Module, path: str):
    """只保存 LoRA 权重（轻量级）"""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state[name] = param.data.clone()
    torch.save(lora_state, path)
    print(f"LoRA 权重已保存到: {path}")


def merge_lora_to_model(model: nn.Module) -> nn.Module:
    """
    将所有 LoRA 权重合并回原始卷积层，并替换为标准 Conv2d

    合并后模型不再包含 Conv2dLoRA 层，可以正常用 YOLO() 加载。
    合并公式: W_merged = W_original + (lora_B @ lora_A) * (alpha / r)
    """
    modules_to_replace = []

    for name, module in model.named_modules():
        if isinstance(module, Conv2dLoRA):
            # 合并权重
            module.merge_weights()
            # 记录需要替换的模块
            modules_to_replace.append((name, module.conv))

    # 将 Conv2dLoRA 替换回标准 Conv2d
    for name, original_conv in modules_to_replace:
        name_parts = name.split('.')
        parent = model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        child_name = name_parts[-1]
        setattr(parent, child_name, original_conv)

    print(f"[LoRA] 已合并 {len(modules_to_replace)} 个 LoRA 层到原始卷积")
    return model


def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """加载 LoRA 权重"""
    lora_state = torch.load(path, map_location='cpu')
    state_dict = model.state_dict()
    
    for name, param in lora_state.items():
        if name in state_dict:
            state_dict[name].copy_(param)
    
    model.load_state_dict(state_dict, strict=False)
    print(f"LoRA 权重已从 {path} 加载")
    return model
