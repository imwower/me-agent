from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.models.multitask_model import MultiTaskModel
from src.constraints.constrained_decode import constrained_generate_from_evidence


def evidence_first_generate(
    image: Any,
    question: str,
    evidence: Dict[str, Any],
    *,
    abstain_threshold: float = 0.4,
    pointer_min_conf: float = 0.35,
    model: Optional[MultiTaskModel] = None,
) -> Dict[str, Any]:
    """基于“证据先行 + 受约束解码”的简化推理流程。

    说明：
        - 当前实现仅依赖证据池本身，不调用大型语言模型；
        - 通过 constrained_generate_from_evidence 从 OCR/图表/表格证据中
          选择最合适的候选答案，并根据置信度决定是否拒答。
    若提供训练好的 MultiTaskModel，则会额外调用其可答性 Head 估计
    「该问题是否值得作答」的概率，并与证据指针置信度结合：
        combined_conf = min(pointer_conf, answerability_prob)
    最终拒答由 combined_conf 与 abstain_threshold 决定。
    """

    # 1) 基于证据池执行约束式指针生成
    answer, abstain_rule, pointer_conf, used_evidence = constrained_generate_from_evidence(
        question=question,
        evidence=evidence,
        abstain_threshold=abstain_threshold,
        pointer_min_conf=pointer_min_conf,
        vocab_trie=None,
    )

    # 2) 若未提供多任务模型，则直接使用指针置信度与规则拒答
    if model is None:
        return {
            "answer": answer,
            "abstain": abstain_rule,
            "confidence": float(pointer_conf),
            "evidence": used_evidence,
        }

    # 3) 使用可答性 Head 估计 answerable 概率
    try:
        answerability_prob = model.predict_answerability(question, image=image)
    except Exception:
        # 若模型推理失败，则退回到规则指针行为
        return {
            "answer": answer,
            "abstain": abstain_rule,
            "confidence": float(pointer_conf),
            "evidence": used_evidence,
        }

    # 综合置信度：既要证据可靠，也要模型认为值得作答
    combined_conf = float(min(pointer_conf, answerability_prob))
    abstain = combined_conf < abstain_threshold

    return {
        "answer": answer,
        "abstain": abstain,
        "confidence": combined_conf,
        "evidence": used_evidence,
        # 可选内部信息，便于调试与评测（不会影响既有消费方）
        "meta": {
            "pointer_conf": float(pointer_conf),
            "answerability_prob": float(answerability_prob),
        },
    }
