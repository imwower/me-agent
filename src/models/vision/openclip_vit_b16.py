from __future__ import annotations

import logging
from typing import List

import open_clip
import torch
from torch import nn

logger = logging.getLogger(__name__)


class OpenCLIPVisionEncoder(nn.Module):
    """ViT-B/16 (OpenCLIP) 视觉编码器封装。

    说明：
        - 默认加载 open_clip 的 ViT-B/16 权重；
        - 仅暴露 encode_image 接口，返回图像全局 embedding；
        - 模型在训练阶段通常冻结，仅作为特征提取器使用。
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        logger.info(
            "加载 OpenCLIP 视觉编码器: model=%s, pretrained=%s, device=%s",
            model_name,
            pretrained,
            self.device,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """编码一批预处理后的图像张量，返回 embedding。"""

        with torch.no_grad():
            feats = self.model.encode_image(images.to(self.device))
        return feats

    def encode_pil_images(self, pil_images: List["object"]) -> torch.Tensor:
        """接收 PIL.Image 列表，完成预处理与编码。"""

        imgs = [self.preprocess(img) for img in pil_images]
        batch = torch.stack(imgs, dim=0).to(self.device)
        return self.forward(batch)

