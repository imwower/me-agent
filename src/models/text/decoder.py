from __future__ import annotations

import logging
from typing import List

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class TextDecoder(nn.Module):
    """基于 HuggingFace Transformers 的中文文本解码器封装。

    默认使用一个中小规模的中文 GPT2 风格模型，仅在桥接层/头部上做轻量微调，
    当前模块主要提供 encode_context / generate 两类接口。
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 64,
    ) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device(device)
        self.model.to(self.device)

        self.max_length = max_length

        logger.info("加载文本解码器模型: %s, device=%s", model_name, self.device)

    def encode(self, texts: List[str]) -> torch.Tensor:
        """将文本编码为 hidden states（用于下游头部/桥接层）。"""

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model.transformer(**enc)  # type: ignore[attr-defined]
        hidden_states = outputs.last_hidden_state
        return hidden_states

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
    ) -> str:
        """根据给定提示生成文本，用于推理阶段回答问题。"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text

