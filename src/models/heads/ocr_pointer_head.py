from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class OCRPointerHead(nn.Module):
    """指针式 OCR 拷贝头。

    输入：
        - decoder_hidden: (batch, hidden_dim) 解码器当前时间步隐藏状态；
        - ocr_hidden: (batch, num_tokens, hidden_dim) OCR token 的表示。
    输出：
        - pointer_logits: (batch, num_tokens) 指向每个 OCR token 的分数。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        ocr_hidden: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_proj(decoder_hidden).unsqueeze(1)  # (B,1,H)
        k = self.key_proj(ocr_hidden)  # (B,N,H)
        # 简单点积注意力
        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1)  # (B,N)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_indices: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """计算指针选择的交叉熵损失。"""

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        return loss_fct(logits, target_indices)

