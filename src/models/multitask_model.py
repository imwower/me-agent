from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.models.bridges.qformer import SimpleQFormerBridge
from src.models.heads.answerability_head import AnswerabilityHead
from src.models.heads.chart_head import ChartHead
from src.models.heads.ocr_pointer_head import OCRPointerHead
from src.models.heads.vqa_head import VQAHead
from src.models.vision.openclip_vit_b16 import OpenCLIPVisionEncoder

logger = logging.getLogger(__name__)


class MultiTaskModel(nn.Module):
    """多任务多模态模型骨架。

    设计目标：
        - 将视觉编码器、桥接层、文本解码器与多任务 head 统一封装；
        - 支持 VQA / OCR-VQA / Chart-QA 等任务共享一套 backbone；
        - 当前实现主要作为结构骨架，训练时仍可仅使用文本侧损失，
          后续逐步启用视觉与 pointer 相关分支。

    注意：
        - 为了避免在导入模块时就下载大模型，建议仅在训练脚本中实例化；
        - 本类不处理优化器与调度器，这些由 training.optimizer 管理。
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.device = device

        model_cfg = model_cfg.get("model", {})

        # 1) 视觉编码器（OpenCLIP），默认冻结，仅用于特征提取
        vision_cfg = model_cfg.get("vision_encoder", {}) or {}
        vision_name = vision_cfg.get("name", "open_clip_vit_b16")
        if vision_name != "open_clip_vit_b16":
            logger.warning("当前 MultiTaskModel 仅针对 open_clip_vit_b16 做了封装。")
        vision_pretrained = vision_cfg.get("pretrained", "openai")
        vision_embed_dim = int(vision_cfg.get("embed_dim", 512))

        self.vision_encoder = OpenCLIPVisionEncoder(
            model_name="ViT-B-16",
            pretrained=vision_pretrained,
            device=str(device),
        )
        self.vision_dim = vision_embed_dim

        # 2) Q-Former 风格桥接层
        bridge_cfg = model_cfg.get("bridge", {}) or {}
        bridge_hidden = int(bridge_cfg.get("hidden_dim", 512))
        bridge_num_queries = int(bridge_cfg.get("num_query_tokens", 32))

        self.bridge = SimpleQFormerBridge(
            vision_dim=self.vision_dim,
            hidden_dim=bridge_hidden,
            num_query_tokens=bridge_num_queries,
        )

        # 3) 文本解码器（当前使用 AutoModelForCausalLM）
        text_cfg = model_cfg.get("text_decoder", {}) or {}
        text_model_name = text_cfg.get("model_name", "gpt2")
        self.max_length = int(text_cfg.get("max_length", 64))

        logger.info("MultiTaskModel: 加载文本模型 %s", text_model_name)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(text_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.text_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(text_model_name)
        self.text_model.to(self.device)

        hidden_dim = self.text_model.config.hidden_size  # type: ignore[attr-defined]
        vocab_size = self.text_model.config.vocab_size  # type: ignore[attr-defined]

        # 4) 多任务头部
        heads_cfg = model_cfg.get("heads", {}) or {}
        vqa_hidden = int(heads_cfg.get("vqa", {}).get("hidden_dim", hidden_dim))
        ans_hidden = int(heads_cfg.get("answerability", {}).get("hidden_dim", hidden_dim // 2))
        ocr_hidden = int(heads_cfg.get("ocr_pointer", {}).get("hidden_dim", hidden_dim // 2))
        chart_hidden = int(heads_cfg.get("chart", {}).get("hidden_dim", hidden_dim // 2))

        self.vqa_head = VQAHead(hidden_dim=vqa_hidden, vocab_size=vocab_size)
        self.answerability_head = AnswerabilityHead(hidden_dim=ans_hidden)
        self.ocr_head = OCRPointerHead(hidden_dim=ocr_hidden)
        self.chart_head = ChartHead(hidden_dim=chart_hidden)

        logger.info(
            "MultiTaskModel 初始化完成: vision=%s, text=%s, hidden_dim=%d",
            vision_name,
            text_model_name,
            hidden_dim,
        )

    # ====================== 文本侧辅助接口 ======================

    def encode_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """将文本批次编码为输入张量。"""

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def lm_forward(self, texts: List[str]) -> torch.Tensor:
        """对一批文本执行自回归语言模型前向，并返回 loss。"""

        enc = self.encode_texts(texts)
        outputs = self.text_model(**enc, labels=enc["input_ids"])
        return outputs.loss  # type: ignore[no-any-return]

    # ====================== 视觉侧辅助接口 ======================

    def encode_images(self, pil_images: List["object"]) -> torch.Tensor:
        """使用 OpenCLIP 对图像进行编码，返回 (batch, vision_dim) 特征。"""

        with torch.no_grad():
            feats = self.vision_encoder.encode_pil_images(pil_images)
        return feats

    def bridge_vision(self, vision_feats: torch.Tensor) -> torch.Tensor:
        """将视觉特征通过桥接层投影到 query token 空间。"""

        return self.bridge(vision_feats)

    # ====================== 任务级接口占位 ======================

    def forward_vqa(
        self,
        texts: List[str],
        images: Optional[List["object"]] = None,
    ) -> torch.Tensor:
        """VQA 任务前向接口（占位实现）。

        当前实现：
            - 仅使用纯文本语言模型 loss；
            - 视觉与桥接层尚未融入实际训练，只作为扩展位存在。
        """

        return self.lm_forward(texts)

    # 后续可以添加 forward_ocr / forward_chart 等方法，
    # 并在其中结合 OCRPointerHead 与 ChartHead 的输出与损失。

