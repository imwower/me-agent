from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def build_batch_texts_from_stream(
    stream_iter: Iterable[Dict[str, object]],
    batch_size: int,
) -> List[str]:
    """从样本流中构造一批用于语言模型训练的文本。

    约定：
        - 每个样本为统一 schema 的字典，至少包含 question/answers 字段；
        - 文本格式采用简单的“问题/答案”拼接：
              问题：{question}\n答案：{answer}
        - 若某条样本缺少答案，则跳过。
    """

    texts: List[str] = []
    it = iter(stream_iter)

    while len(texts) < batch_size:
        try:
            ex = next(it)
        except StopIteration:
            break

        q = ex.get("question") or ""
        ans_list = ex.get("answers") or []
        if not ans_list:
            continue
        a = ans_list[0]
        text = f"问题：{q}\n答案：{a}"
        texts.append(text)

    return texts


def compute_lm_step_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    *,
    device: torch.device,
    max_length: int,
    grad_accum_steps: int,
) -> Tuple[torch.Tensor, float]:
    """对一批文本计算语言模型自回归损失。

    返回：
        loss: 已按 grad_accum_steps 缩放后的 tensor，用于 backward；
        scalar_loss: 该批次的标量损失值（未缩放或已缩放均可，当前为缩放后）。
    """

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    outputs = model(**enc, labels=enc["input_ids"])
    loss = outputs.loss / grad_accum_steps
    return loss, float(loss.item())

