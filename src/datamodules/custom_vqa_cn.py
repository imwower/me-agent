from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .base import UnifiedSample

logger = logging.getLogger(__name__)


def _convert_custom_example(ex: Dict[str, Any]) -> UnifiedSample:
    """将自定义 JSONL VQA 样本转换为 UnifiedSample。

    约定输入格式（每行 JSON）：
        {
          "image": "<可选，图像路径或 URL>",
          "question": "<中文问题>",
          "answer": "<答案字符串>",
          "answers": ["<答案1>", "<答案2>", ...],  # 可选，若存在则优先使用
          "answerable": true/false,              # 可选
          "meta": {...}                          # 可选
        }
    """

    image = ex.get("image")
    question = str(ex.get("question") or "")

    answers: List[str]
    if "answers" in ex and isinstance(ex["answers"], list):
        answers = [str(a) for a in ex["answers"] if a is not None]
    else:
        ans = ex.get("answer")
        answers = [str(ans)] if ans is not None else []

    answerable = ex.get("answerable")
    if not isinstance(answerable, bool):
        answerable = None

    meta = ex.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}

    unified = UnifiedSample(
        image=image,
        question=question,
        answers=answers,
        answerable=answerable,
        evidence={
            "ocr_tokens": [],
            "regions": [],
            "chart_elements": [],
        },
        meta=meta,
    )
    return unified


def load_custom_vqa_cn(
    path: str = "data/vqa_cn/custom_vqa_cn.jsonl",
    sample_ratio: float = 1.0,
) -> Iterable[UnifiedSample]:
    """从本地 JSONL 文件加载自定义中文 VQA / 文本 QA 样本。

    参数：
        path: JSONL 文件路径；
        sample_ratio: 抽样比例（0,1]，用于快速实验。
    """

    p = Path(path)
    if not p.exists():
        logger.warning("未找到自定义 VQA 数据文件: %s，将返回空样本流。", p)
        return []

    logger.info("加载自定义 VQA 数据集: %s, sample_ratio=%.3f", p, sample_ratio)

    def _iter() -> Iterable[UnifiedSample]:
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if sample_ratio < 1.0 and (i % int(1.0 / sample_ratio)) != 0:
                    continue
                try:
                    ex = json.loads(line)
                except Exception:
                    continue
                yield _convert_custom_example(ex)

    return _iter()

