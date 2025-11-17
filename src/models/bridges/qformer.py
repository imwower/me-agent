from __future__ import annotations

import logging
from typing import Tuple

import torch
from torch import nn

from src.models.utils.lora_adapter import LoRALinear

logger = logging.getLogger(__name__)


class SimpleQFormerBridge(nn.Module):
    """简化版 Q-Former 风格桥接层。

    设计说明：
        - 输入为视觉编码器输出的全局图像 embedding (batch, D_v)；
        - 内部维护若干可学习的 query token (num_query_tokens, D_q)；
        - 使用多层 Transformer 编码器将视觉 embedding 作为 Key/Value，
          query token 作为 Query，得到 (batch, num_query_tokens, D_q)；
        - 输出作为文本解码器的先验条件或附加上下文。

    当前实现：
        - 为保持简单，仅使用单层多头注意力 + 前馈；
        - 仍保留 num_query_tokens / hidden_dim 等配置，便于后续替换为真正 Q-Former。
    """

    def __init__(
        self,
        vision_dim: int,
        hidden_dim: int = 512,
        num_query_tokens: int = 32,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
    ) -> None:
        super().__init__()

        self.num_query_tokens = num_query_tokens

        self.query_tokens = nn.Parameter(
            torch.randn(num_query_tokens, hidden_dim) * 0.02
        )

        # 视觉投影层可以启用 LoRA，以便在不修改预训练编码器的情况下微调桥接层。
        if use_lora:
            self.vision_proj = LoRALinear(
                in_features=vision_dim,
                out_features=hidden_dim,
                r=lora_r,
                alpha=lora_alpha,
            )
            logger.info(
                "SimpleQFormerBridge: 使用 LoRALinear 作为 vision_proj (r=%d, alpha=%.1f)",
                lora_r,
                lora_alpha,
            )
        else:
            self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        logger.info(
            "初始化 SimpleQFormerBridge: vision_dim=%d, hidden_dim=%d, num_query_tokens=%d",
            vision_dim,
            hidden_dim,
            num_query_tokens,
        )

    def forward(self, vision_feats: torch.Tensor) -> torch.Tensor:
        """将视觉特征投影并通过 query 机制生成跨模态特征。

        参数：
            vision_feats: (batch, D_v) 视觉编码器输出的全局特征。

        返回：
            query_out: (batch, num_query_tokens, hidden_dim)
        """

        bsz = vision_feats.size(0)
        vision_proj = self.vision_proj(vision_feats)  # (batch, hidden_dim)
        vision_proj = vision_proj.unsqueeze(1)  # (batch, 1, hidden_dim)

        # 扩展 query tokens 到 batch 维度
        query = self.query_tokens.unsqueeze(0).expand(bsz, -1, -1)  # (batch, num_q, hidden_dim)

        # 自注意力：query 查询 vision_proj（作为 key/value）
        attn_output, _ = self.self_attn(
            query,  # query
            vision_proj,  # key
            vision_proj,  # value
        )
        out = self.norm1(query + attn_output)

        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)
        return out
