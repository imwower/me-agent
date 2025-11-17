from __future__ import annotations

from typing import Iterable, List

from .metrics import compute_vqa_accuracy  # 复用已有实现


def vqa_accuracy(
    preds: Iterable[str],
    refs: Iterable[List[str]],
) -> float:
    """中文 VQA 准确率指标包装。

    目前直接调用通用 compute_vqa_accuracy，内部已做简单归一化：
        - 去首尾空格；
        - 转小写。
    后续可在此处添加中文全角/半角与单位归一逻辑。
    """

    return compute_vqa_accuracy(preds, refs)

