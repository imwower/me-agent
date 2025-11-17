from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .pointer_copy import select_best_token
from .trie import Trie


def constrained_generate_from_evidence(
    question: str,
    evidence: Dict[str, Any],
    *,
    abstain_threshold: float = 0.4,
    pointer_min_conf: float = 0.35,
    vocab_trie: Trie | None = None,
) -> Tuple[str, bool, float, List[Dict[str, Any]]]:
    """基于证据池和简单约束生成答案。

    返回：
        answer: 生成的答案文本（可能为空字符串）；
        abstain: 是否选择拒答；
        confidence: 对当前答案的置信度估计 [0,1]；
        used_evidence: 一个列表，记录参与决策的证据条目（含 type/id/confidence 等）。

    说明：
        - 该实现不依赖具体语言模型，仅通过启发式规则实现“证据先行 + 受约束解码”；
        - 在真实系统中，可以将本函数替换为融合 LM logits 和 pointer 分布的 beam search。
    """

    ocr_tokens: List[Dict[str, Any]] = list(evidence.get("ocr_tokens") or [])
    chart_elements: List[Dict[str, Any]] = list(evidence.get("chart_elements") or [])
    table_cells: List[Dict[str, Any]] = list(evidence.get("table_cells") or [])

    used: List[Dict[str, Any]] = []
    best_conf = 0.0
    answer = ""

    # 1) 先尝试从表格单元格中抽取（优先结构化字段）
    if table_cells:
        best_cell, conf = select_best_token(table_cells, question)
        if best_cell is not None and conf >= pointer_min_conf:
            text = str(best_cell.get("text") or best_cell.get("value") or "")
            answer = text
            best_conf = conf
            used.append(
                {
                    "type": "cell",
                    "id": best_cell.get("id"),
                    "text": text,
                    "confidence": conf,
                }
            )

    # 2) 其次尝试从图表元素中抽取数值/标签
    if not answer and chart_elements:
        best_elem, conf = select_best_token(chart_elements, question)
        if best_elem is not None and conf >= pointer_min_conf:
            meta = best_elem.get("meta") or {}
            # 按优先级选择一个可读字段作为答案
            text = (
                str(meta.get("y"))
                if meta.get("y") is not None
                else str(meta.get("value") or meta.get("x") or "")
            )
            answer = text
            best_conf = conf
            used.append(
                {
                    "type": "chart_elem",
                    "id": best_elem.get("id"),
                    "text": text,
                    "confidence": conf,
                }
            )

    # 3) 最后尝试从 OCR tokens 中抽取
    if not answer and ocr_tokens:
        best_tok, conf = select_best_token(ocr_tokens, question)
        if best_tok is not None and conf >= pointer_min_conf:
            text = str(best_tok.get("text") or "")
            answer = text
            best_conf = conf
            used.append(
                {
                    "type": "ocr",
                    "id": best_tok.get("id"),
                    "text": text,
                    "confidence": conf,
                }
            )

    # 4) 如果有 Trie 约束，则验证答案是否满足前缀词典
    if vocab_trie is not None and answer:
        # 若答案不在词典中，略微降低置信度
        if not vocab_trie.match_prefix(answer):
            best_conf *= 0.7

    if not answer:
        # 完全找不到答案候选
        return "", True, 0.0, used

    # 5) 根据置信度决定是否拒答
    abstain = best_conf < abstain_threshold
    return answer, abstain, best_conf, used

