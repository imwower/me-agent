from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """简单的 LoRA 版 Linear 层实现。

    设计目标：
        - 兼容 nn.Linear 的前向接口；
        - 保持原始权重为冻结，仅通过低秩矩阵 (A, B) 学习增量；
        - 适合作为桥接层或部分 Head 上的小型微调适配器。

    数学形式：
        y = x W^T + scale * x A^T B^T + b
    其中：
        - W 为预训练权重（冻结）；
        - A, B 为可训练的低秩矩阵；
        - scale = alpha / r 为缩放系数。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r) if r > 0 else 1.0

        # 冻结的基础权重，相当于预训练 Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.bias.requires_grad = False
        else:
            self.bias = None

        # LoRA 低秩增量参数
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        else:
            # 当 r=0 时，相当于不启用 LoRA
            self.lora_A = None
            self.lora_B = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化基础权重与 LoRA 权重。"""

        # 基础权重使用与 nn.Linear 类似的初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)  # type: ignore[attr-defined]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # LoRA 增量参数通常初始化为 0，使得初始时等价于原始权重
        if self.lora_A is not None and self.lora_B is not None:
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        result = torch.nn.functional.linear(x, self.weight, self.bias)

        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            # x: (..., in), A: (r, in), B: (out, r)
            lora_out = torch.nn.functional.linear(x, self.lora_A)
            lora_out = torch.nn.functional.linear(lora_out, self.lora_B)
            result = result + self.scaling * lora_out

        return result


def only_lora_parameters(mod: nn.Module) -> Iterable[nn.Parameter]:
    """从模块中筛选出 LoRA 参数（名称包含 'lora_' 的参数）。

    用于在需要时只对 LoRA 增量进行优化，而冻结其他权重。
    """

    for name, param in mod.named_parameters():
        if "lora_" in name:
            yield param

