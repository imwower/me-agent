from __future__ import annotations

import torch
from typing import Dict, List


def evaluate_recall_at_k(text_emb: torch.Tensor, image_emb: torch.Tensor, ks: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    简易检索评估：计算文本->图像、图像->文本的 Recall@K。
    """

    text_emb = torch.nn.functional.normalize(text_emb, dim=-1)
    image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
    sim = text_emb @ image_emb.t()
    results: Dict[str, float] = {}
    n = sim.size(0)
    for k in ks:
        kk = min(k, n)
        top_text = sim.topk(k=kk, dim=1).indices
        correct_text = torch.eq(top_text, torch.arange(sim.size(0), device=sim.device).unsqueeze(1)).any(dim=1).float().mean()
        top_img = sim.topk(k=kk, dim=0).indices
        correct_img = torch.eq(top_img, torch.arange(sim.size(1), device=sim.device).unsqueeze(0)).any(dim=0).float().mean()
        results[f"t2i_recall@{k}"] = float(correct_text)
        results[f"i2t_recall@{k}"] = float(correct_img)
    return results
