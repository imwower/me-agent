from __future__ import annotations

from typing import Iterable, Tuple

from .metrics import compute_ocr_metrics  # 复用已有实现


def ocr_vqa_scores(
    preds: Iterable[str],
    refs: Iterable[str],
) -> Tuple[float, float]:
    """OCR-VQA/DocVQA 指标包装。

    返回：
        hit_rate: 答案来自 OCR 词表的命中率近似；
        avg_edit_distance: 平均字符级编辑距离。
    """

    return compute_ocr_metrics(preds, refs)

