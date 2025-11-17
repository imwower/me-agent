from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

from .base import UnifiedSample

logger = logging.getLogger(__name__)


def _encode_image_field(example: Dict[str, Any]) -> Any:
    """将 Chinese-SimpleVQA 的图像字段转换为统一 image 字段。

    该数据集在 HuggingFace 上的字段通常包含 base64 / URL / image 对象等，
    为了兼容 streaming 模式，这里倾向于直接保留原始字段（例如 URL 或 PIL Image）。
    具体落地策略可根据需要在后续版本中扩展。
    """

    # 若存在 "image" 字段且为 image 对象，直接返回
    if "image" in example:
        return example["image"]

    # 若存在 "image_path" 或 "image_url" 等字段，则直接返回该路径/URL
    for key in ("image_path", "image_url", "url"):
        if key in example:
            return example[key]

    return None


def convert_example(example: Dict[str, Any]) -> UnifiedSample:
    """将 Chinese-SimpleVQA 的一条样本转换为统一 schema。"""

    image = _encode_image_field(example)
    question = str(example.get("final_question") or example.get("question") or "")
    answer = str(example.get("final_answer") or example.get("answer") or "")

    answers: List[str] = [answer] if answer else []

    unified = UnifiedSample(
        image=image,
        question=question,
        answers=answers,
        answerable=None,  # 该数据集未显式提供可答性标签
        evidence={
            "ocr_tokens": [],
            "regions": [],
            "chart_elements": [],
        },
        meta={
            "dataset": "csVQA",
            "split": "unknown",
        },
    )
    return unified


def load_chinese_simplevqa(
    split: str = "train",
    sample_ratio: float = 1.0,
    cache_dir: Optional[str] = None,
) -> Iterable[UnifiedSample]:
    """以 streaming 模式加载 Chinese-SimpleVQA 并转换为 UnifiedSample 序列。

    参数：
        split: 使用的数据集切分（train/validation/test 等）；
        sample_ratio: 抽样比例（0,1]，用于在本地快速跑通实验；
        cache_dir: 本地缓存目录。
    """

    logger.info(
        "加载 Chinese-SimpleVQA 数据集: split=%s, sample_ratio=%.3f",
        split,
        sample_ratio,
    )

    ds = load_dataset(
        "OpenStellarTeam/Chinese-SimpleVQA",
        split=split,
        streaming=True,
        cache_dir=cache_dir,
    )

    for i, example in enumerate(ds):
        # 抽样：简单使用下标 + sample_ratio 控制
        if sample_ratio < 1.0:
            if (i % int(1.0 / sample_ratio)) != 0:
                continue
        yield convert_example(example)

