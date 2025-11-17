from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

from src.charts.parser import parse_chart_metadata
from src.datamodules.base import ChartElement, UnifiedSample

logger = logging.getLogger(__name__)


def convert_chartqa_example(example: Dict[str, Any]) -> UnifiedSample:
    """将 ChartQA 的一条样本转换为统一 schema。

    说明：
        - ChartQA 官方格式可能包含 chart 类型、数据表等信息；
        - 此处假设存在字段：
            * image: 图像对象或路径；
            * question: 问题文本；
            * answer: 答案文本或数值；
            * meta_path: 指向图表元数据文件的可选路径。
        - 若缺少 meta_path，则仅构造空的 chart_elements。
    """

    image = example.get("image")
    question = str(example.get("question") or "")
    answer_raw = example.get("answer")
    if isinstance(answer_raw, (int, float)):
        answer_str = str(answer_raw)
    else:
        answer_str = str(answer_raw or "")

    answers: List[str] = [answer_str] if answer_str else []

    chart_elements: List[Dict[str, Any]] = []
    meta_path = example.get("meta_path")
    if isinstance(meta_path, str) and meta_path:
        try:
            chart_elements = parse_chart_metadata(meta_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("解析 ChartQA 元数据失败: %s", exc)

    unified = UnifiedSample(
        image=image,
        question=question,
        answers=answers,
        answerable=True if answers else None,
        evidence={
            "ocr_tokens": [],
            "regions": [],
            "chart_elements": chart_elements,
            "table_cells": [],
        },
        meta={
            "dataset": "ChartQA",
            "split": "unknown",
        },
    )
    return unified


def load_chartqa(
    split: str = "train",
    sample_ratio: float = 1.0,
    cache_dir: Optional[str] = None,
) -> Iterable[UnifiedSample]:
    """以 streaming 模式加载 ChartQA 并转换为 UnifiedSample 序列。

    说明：
        - ChartQA 数据集可通过 HuggingFace Datasets 获取（若网络可用）；
        - 若加载失败，调用方将收到一个空迭代器。
    """

    logger.info(
        "尝试通过 HuggingFace Datasets 加载 ChartQA: split=%s, sample_ratio=%.3f",
        split,
        sample_ratio,
    )

    try:
        ds = load_dataset(
            "chartqa",
            split=split,
            streaming=True,
            cache_dir=cache_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "加载 ChartQA 数据集失败: %s\n"
            "请在本地下载数据集并根据 README 将其放置到 data/chartqa 下，"
            "或者在后续迭代中扩展本函数读取本地索引。",
            exc,
        )
        return []

    for i, example in enumerate(ds):
        if sample_ratio < 1.0:
            if (i % int(1.0 / sample_ratio)) != 0:
                continue
        yield convert_chartqa_example(example)

