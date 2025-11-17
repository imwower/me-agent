from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

from .base import OcrToken, UnifiedSample

logger = logging.getLogger(__name__)


def _build_ocr_tokens(document: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """从 XFUND-zh 的 document 字段构造统一 OCR token 列表。

    说明：
        - 该实现假设每个 token 具有 text 与 bbox 字段；
        - bbox 一般为 [x1, y1, x2, y2]，若格式不同可在此处进行适配；
        - 为简化实现，id 直接使用索引。
    """

    tokens: List[Dict[str, Any]] = []
    for idx, tok in enumerate(document):
        text = str(tok.get("text") or "")
        bbox = tok.get("bbox") or [0, 0, 0, 0]
        if not isinstance(bbox, list) or len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        tokens.append(
            OcrToken(
                id=f"t{idx}",
                text=text,
                bbox=[int(v) for v in bbox],
            ).__dict__
        )
    return tokens


def convert_example(example: Dict[str, Any]) -> UnifiedSample:
    """将 XFUND-zh 的一条样本转换为统一 schema。

    简化设定：
        - question: 使用模板“这张文档的主要标题是什么？”；
        - answers: 取 document 中前若干 token 连接而成，作为占位答案；
        - 实际使用时建议根据实体标注构造更精确的问答对。
    """

    image = example.get("image")  # HF datasets 通常会自动解码为 PIL.Image
    document = example.get("document") or []

    ocr_tokens = _build_ocr_tokens(document)

    # 简单地将前若干 token 作为占位答案（真实任务可改为发票号/金额等字段）
    answer_tokens = [t["text"] for t in ocr_tokens[:5] if t["text"].strip()]
    answer_text = "".join(answer_tokens)

    unified = UnifiedSample(
        image=image,
        question="这张文档的主要内容是什么？",
        answers=[answer_text] if answer_text else [],
        answerable=True if answer_text else None,
        evidence={
            "ocr_tokens": ocr_tokens,
            "regions": [],
            "chart_elements": [],
        },
        meta={
            "dataset": "xfund_zh",
            "split": "unknown",
        },
    )
    return unified


def load_xfund_zh(
    split: str = "train",
    sample_ratio: float = 1.0,
    cache_dir: Optional[str] = None,
) -> Iterable[UnifiedSample]:
    """以 streaming 模式加载 XFUND-zh 并转换为 UnifiedSample 序列。"""

    logger.info(
        "加载 XFUND-zh 数据集: split=%s, sample_ratio=%.3f",
        split,
        sample_ratio,
    )

    ds = load_dataset(
        "xfund",
        "zh",
        split=split,
        streaming=True,
        cache_dir=cache_dir,
    )

    for i, example in enumerate(ds):
        if sample_ratio < 1.0:
            if (i % int(1.0 / sample_ratio)) != 0:
                continue
        yield convert_example(example)

