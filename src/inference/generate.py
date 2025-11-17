from __future__ import annotations

from typing import Any, Dict, Tuple

from src.constraints.constrained_decode import constrained_generate_from_evidence


def evidence_first_generate(
    image: Any,
    question: str,
    evidence: Dict[str, Any],
    *,
    abstain_threshold: float = 0.4,
    pointer_min_conf: float = 0.35,
) -> Dict[str, Any]:
    """基于“证据先行 + 受约束解码”的简化推理流程。

    说明：
        - 当前实现仅依赖证据池本身，不调用大型语言模型；
        - 通过 constrained_generate_from_evidence 从 OCR/图表/表格证据中
          选择最合适的候选答案，并根据置信度决定是否拒答。
    """

    answer, abstain, confidence, used_evidence = constrained_generate_from_evidence(
        question=question,
        evidence=evidence,
        abstain_threshold=abstain_threshold,
        pointer_min_conf=pointer_min_conf,
        vocab_trie=None,
    )

    return {
        "answer": answer,
        "abstain": abstain,
        "confidence": confidence,
        "evidence": used_evidence,
    }

