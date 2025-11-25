from __future__ import annotations

import torch
from torch import nn


class MultimodalBackbone(nn.Module):
    """
    轻量多模态 backbone 占位：使用 stub 特征 + 可训练投影层。
    """

    def __init__(self, input_dim: int = 256, proj_dim: int = 128, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.freeze_backbone = freeze_backbone
        self.text_proj = nn.Linear(input_dim, proj_dim, bias=False)
        self.vision_proj = nn.Linear(input_dim, proj_dim, bias=False)

    def encode_text(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.text_proj(feats)
        return torch.nn.functional.normalize(x, dim=-1)

    def encode_image(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.vision_proj(feats)
        return torch.nn.functional.normalize(x, dim=-1)

    def forward(self, text_feats: torch.Tensor, image_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_text(text_feats), self.encode_image(image_feats)
