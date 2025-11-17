from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from sklearn.metrics import average_precision_score


def _normalize_answer(ans: str) -> str:
    """对答案做简单归一化，便于 VQA-Acc 计算。"""

    return ans.strip().lower()


def compute_vqa_accuracy(
    preds: Iterable[str],
    refs: Iterable[List[str]],
) -> float:
    """计算简化版 VQA-Acc：精确匹配或同义归一后匹配。"""

    correct = 0
    total = 0
    for pred, ref_list in zip(preds, refs):
        total += 1
        norm_pred = _normalize_answer(pred)
        norm_refs = {_normalize_answer(r) for r in ref_list}
        if norm_pred in norm_refs:
            correct += 1
    return correct / total if total > 0 else 0.0


def compute_answerability_ap(
    scores: Iterable[float],
    labels: Iterable[int],
) -> float:
    """计算可答性二分类的 Average Precision。"""

    y_scores = list(scores)
    y_true = list(labels)
    if not y_true:
        return 0.0
    return float(average_precision_score(y_true, y_scores))


def _edit_distance(a: str, b: str) -> int:
    """简单的编辑距离实现，用于 OCR 文本评估。"""

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def compute_ocr_metrics(
    preds: Iterable[str],
    refs: Iterable[str],
) -> Tuple[float, float]:
    """计算 OCR-VQA 相关指标：命中率与平均编辑距离。

    返回：
        (hit_rate, avg_edit_distance)
    """

    hits = 0
    total = 0
    dist_sum = 0

    for pred, ref in zip(preds, refs):
        total += 1
        if pred.strip() and pred.strip() in ref:
            hits += 1
        dist_sum += _edit_distance(pred, ref)

    if total == 0:
        return 0.0, 0.0
    return hits / total, dist_sum / total


def compute_chart_metrics(
    pred_values: Iterable[float],
    true_values: Iterable[float],
) -> Tuple[float, float]:
    """计算图表任务的 EM 与数值误差（MAPE）。"""

    em_correct = 0
    total = 0
    ape_sum = 0.0

    for pred, true in zip(pred_values, true_values):
        total += 1
        if math.isclose(pred, true, rel_tol=1e-6, abs_tol=1e-6):
            em_correct += 1
        if true != 0:
            ape_sum += abs((pred - true) / true)

    if total == 0:
        return 0.0, 0.0

    em = em_correct / total
    mape = ape_sum / total if ape_sum > 0 else 0.0
    return em, mape

