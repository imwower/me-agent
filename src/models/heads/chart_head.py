from __future__ import annotations

import torch
from torch import nn


class ChartHead(nn.Module):
    """图表值抽取头。

    输入：
        - pooled_hidden: (batch, hidden_dim) 多模态聚合向量；
        - chart_hidden: (batch, num_elements, hidden_dim) 图表元素表征。
    输出：
        - logits: (batch, num_elements) 选择目标元素的分数；
        - value_pred: (batch,) 对目标值的数值预测（可选）。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        pooled_hidden: torch.Tensor,
        chart_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_proj(pooled_hidden).unsqueeze(1)  # (B,1,H)
        k = self.key_proj(chart_hidden)  # (B,N,H)
        logits = torch.bmm(q, k.transpose(1, 2)).squeeze(1)  # (B,N)

        # 使用注意力加权图表元素表征，再预测数值
        attn = torch.softmax(logits, dim=-1).unsqueeze(-1)  # (B,N,1)
        context = (attn * chart_hidden).sum(dim=1)  # (B,H)
        value_pred = self.value_regressor(context).squeeze(-1)  # (B,)
        return logits, value_pred

    def compute_classification_loss(
        self,
        logits: torch.Tensor,
        target_indices: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        return loss_fct(logits, target_indices)

    def compute_regression_loss(
        self,
        value_pred: torch.Tensor,
        target_values: torch.Tensor,
    ) -> torch.Tensor:
        loss_fct = nn.L1Loss()
        return loss_fct(value_pred, target_values.float())

