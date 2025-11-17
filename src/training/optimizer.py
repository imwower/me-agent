from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def create_optimizer(
    model_or_params: torch.nn.Module | Iterable[torch.nn.Parameter],
    train_cfg: Dict[str, Any],
) -> AdamW:
    """基于训练配置创建 AdamW 优化器。

    主要参数：
        - lr: 学习率
        - weight_decay: 权重衰减
    """

    lr = float(train_cfg.get("lr", 1.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    if isinstance(model_or_params, nn.Module):
        params = (p for p in model_or_params.parameters() if p.requires_grad)
    else:
        params = model_or_params

    return AdamW(params, lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer: AdamW,
    train_cfg: Dict[str, Any],
    num_training_steps: int,
) -> LambdaLR:
    """创建一个简单的学习率调度器。

    当前实现支持两种模式：
        - linear: 线性衰减到 0；
        - cosine: 余弦退火。
    """

    name = (train_cfg.get("scheduler", {}) or {}).get("name", "linear")

    if num_training_steps <= 0:
        num_training_steps = 1

    def lr_lambda(step: int) -> float:
        progress = step / float(max(1, num_training_steps))
        if name == "cosine":
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265))).item()
        # 默认线性衰减
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0) -> None:
    """对模型参数进行梯度裁剪。"""

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
