from __future__ import annotations

from typing import Iterable, Tuple

from .metrics import compute_chart_metrics  # 复用已有实现


def chart_scores(
    pred_values: Iterable[float],
    true_values: Iterable[float],
) -> Tuple[float, float]:
    """图表问答/数值抽取指标包装。

    返回：
        em: 精确匹配率；
        mape: 平均绝对百分比误差。
    """

    return compute_chart_metrics(pred_values, true_values)

