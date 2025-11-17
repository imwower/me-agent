from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _is_numeric_token(text: str) -> bool:
    """粗略判断一个 token 是否“偏数值”。

    使用简单规则：
        - 含有数字；
        - 或者匹配类似百分比/小数形式。
    """

    if re.search(r"\d", text):
        return True
    if re.search(r"\d+(\.\d+)?%", text):
        return True
    return False


def select_best_token(
    tokens: Iterable[Dict[str, Any]],
    question: str,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """在给定 OCR/表格 token 池中选出最合适的候选。

    这是一个非常简化的“指针网络”替代实现，用于在缺乏完整模型
    的情况下，仍然能体现“从证据池复制”的思想。

    规则：
        - 若问题里包含“多少/几/金额/价格/率/百分比”等关键词：
            * 优先选择包含数字的 token；
            * 在这些 token 中，按 conf 降序，其次按文本长度降序选择；
        - 否则：
            * 在全部 token 中按 conf 与长度排序；
        - 若没有任何 token，则返回 (None, 0.0)。
    """

    tokens_list: List[Dict[str, Any]] = list(tokens)
    if not tokens_list:
        return None, 0.0

    q = question or ""
    need_number = any(k in q for k in ("多少", "几", "金额", "价格", "率", "百分比", "%"))

    def score(tok: Dict[str, Any]) -> Tuple[float, int]:
        text = str(tok.get("text") or "")
        conf = float(tok.get("conf", 0.5))
        return conf, len(text)

    candidates = tokens_list
    if need_number:
        numeric = [t for t in tokens_list if _is_numeric_token(str(t.get("text") or ""))]
        if numeric:
            candidates = numeric

    # 按置信度和长度排序
    candidates.sort(key=score, reverse=True)
    best = candidates[0]
    best_conf = float(best.get("conf", 0.5))

    return best, best_conf

