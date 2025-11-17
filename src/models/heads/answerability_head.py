from __future__ import annotations

import torch
from torch import nn


class AnswerabilityHead(nn.Module):
    """可答性二分类头。

    输入：
        - pooled_hidden: (batch, hidden_dim) 文本/多模态的池化向量。
    输出：
        - logits: (batch, 1) 对应 answerable=True 的 logits。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled_hidden).squeeze(-1)

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """计算二分类 BCEWithLogitsLoss。"""

        loss_fct = nn.BCEWithLogitsLoss()
        return loss_fct(logits, labels.float())

