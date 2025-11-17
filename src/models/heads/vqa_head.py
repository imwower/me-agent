from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class VQAHead(nn.Module):
    """VQA 生成头。

    设计简化版：
        - 输入为文本解码器隐藏状态 (batch, seq_len, hidden_dim)；
        - 输出 logits (batch, seq_len, vocab_size)，供交叉熵损失计算；
        - 实际上可直接复用解码器的语言头，此处单独封装便于多任务拆分。
    """

    def __init__(self, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """计算 teacher forcing 下的交叉熵损失。"""

        vocab_size = logits.size(-1)
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        return loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1),
        )

