from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import MultimodalBackbone
from .eval import evaluate_recall_at_k


@dataclass
class TrainStats:
    loss_history: List[float] = field(default_factory=list)
    eval_history: List[Dict[str, float]] = field(default_factory=list)


class MultimodalTrainer:
    def __init__(
        self,
        model: MultimodalBackbone,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.temp = nn.Parameter(torch.tensor(0.07))
        self.stats = TrainStats()

    def _clip_loss(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        logits_per_text = text_emb @ image_emb.t() / self.temp.exp()
        logits_per_image = image_emb @ text_emb.t() / self.temp.exp()
        targets = torch.arange(text_emb.size(0), device=text_emb.device)
        loss_t = nn.functional.cross_entropy(logits_per_text, targets)
        loss_i = nn.functional.cross_entropy(logits_per_image, targets)
        return (loss_t + loss_i) / 2.0

    def train_clip_style(
        self,
        max_steps: int = 200,
        batch_size: int = 8,
        lr: float = 1e-4,
        log_every: int = 50,
        eval_every: int = 200,
        output_dir: Optional[str] = None,
    ) -> TrainStats:
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)
        step = 0
        while step < max_steps:
            for batch in self.train_loader:
                self.model.train()
                text = batch["text_feats"].to(self.device)
                image = batch["image_feats"].to(self.device)
                text_emb, image_emb = self.model(text, image)
                loss = self._clip_loss(text_emb, image_emb)
                loss.backward()
                optim.step()
                optim.zero_grad(set_to_none=True)
                self.stats.loss_history.append(float(loss.detach()))
                step += 1
                if step % log_every == 0:
                    print(f"[multimodal] step={step} loss={loss:.4f}")  # noqa: T201
                if self.eval_loader and step % eval_every == 0:
                    self._run_eval()
                if step >= max_steps:
                    break
        if self.eval_loader:
            self._run_eval()
        if output_dir:
            self.save(output_dir)
        return self.stats

    def _run_eval(self) -> None:
        if not self.eval_loader:
            return
        self.model.eval()
        with torch.no_grad():
            all_text: List[torch.Tensor] = []
            all_image: List[torch.Tensor] = []
            for batch in self.eval_loader:
                text = batch["text_feats"].to(self.device)
                image = batch["image_feats"].to(self.device)
                t_emb, i_emb = self.model(text, image)
                all_text.append(t_emb)
                all_image.append(i_emb)
            if all_text and all_image:
                text_emb = torch.cat(all_text, dim=0)
                image_emb = torch.cat(all_image, dim=0)
                metrics = evaluate_recall_at_k(text_emb, image_emb)
                self.stats.eval_history.append(metrics)
                print(f"[multimodal] eval {metrics}")  # noqa: T201

    def save(self, output_dir: str) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "text_proj": self.model.text_proj.state_dict(),
            "vision_proj": self.model.vision_proj.state_dict(),
            "temp": float(self.temp.detach().cpu()),
            "input_dim": self.model.input_dim,
            "proj_dim": self.model.proj_dim,
        }
        torch.save(state, path / "multimodal_projection.pt")
