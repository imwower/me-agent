from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import torch
from torch.utils.data import Dataset

try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore


class MultimodalExample(TypedDict, total=False):
    id: str
    image_path: str
    text: str
    labels: List[str]
    task: str


def _load_jsonl(path: Path) -> List[MultimodalExample]:
    examples: List[MultimodalExample] = []
    if not path.exists():
        return examples
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                examples.append(obj)  # type: ignore[arg-type]
        except Exception:
            continue
    return examples


def load_internal_multimodal(path: str) -> List[MultimodalExample]:
    return _load_jsonl(Path(path))


def load_external_multimodal(paths: List[str]) -> List[MultimodalExample]:
    examples: List[MultimodalExample] = []
    for p in paths:
        examples.extend(_load_jsonl(Path(p)))
    return examples


def build_train_eval_splits(
    internal_paths: List[str],
    external_paths: Optional[List[str]] = None,
    eval_ratio: float = 0.1,
    max_samples: int = 50000,
) -> tuple[List[MultimodalExample], List[MultimodalExample]]:
    data: List[MultimodalExample] = []
    for p in internal_paths:
        data.extend(load_internal_multimodal(p))
    for p in external_paths or []:
        data.extend(load_external_multimodal([p]))
    if not data:
        return [], []
    if len(data) > max_samples:
        data = random.sample(data, max_samples)
    random.shuffle(data)
    split = max(1, int(len(data) * (1 - eval_ratio)))
    return data[:split], data[split:]


def _hash_vec(text: str, dim: int) -> torch.Tensor:
    seed = abs(hash(text)) % (2**32)
    rng = random.Random(seed)
    vec = torch.tensor([rng.uniform(-1.0, 1.0) for _ in range(dim)], dtype=torch.float32)
    return torch.nn.functional.normalize(vec, dim=0)


def _encode_text_stub(text: str, dim: int = 256) -> torch.Tensor:
    return _hash_vec(text or "", dim)


def _encode_image_stub(path: str, dim: int = 256) -> torch.Tensor:
    if Image is None:
        return _hash_vec(path or "image", dim)
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((32, 32))
        arr = torch.tensor(list(img.getdata()), dtype=torch.float32).view(-1)
        if arr.numel() < dim:
            pad = torch.zeros(dim - arr.numel())
            arr = torch.cat([arr, pad], dim=0)
        return torch.nn.functional.normalize(arr[:dim], dim=0)
    except Exception:
        return _hash_vec(path or "image", dim)


@dataclass
class MultimodalDataset(Dataset):
    examples: List[MultimodalExample]
    feature_dim: int = 256

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        text = str(ex.get("text") or ex.get("description") or "")
        image_path = str(ex.get("image_path") or "")
        text_feat = _encode_text_stub(text, dim=self.feature_dim)
        image_feat = _encode_image_stub(image_path, dim=self.feature_dim)
        labels = ex.get("labels") or []
        return {
            "text_feat": text_feat,
            "image_feat": image_feat,
            "labels": labels,
            "id": ex.get("id", str(idx)),
        }


def collate_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    text_feats = torch.stack([it["text_feat"] for it in items])
    image_feats = torch.stack([it["image_feat"] for it in items])
    labels = [it.get("labels") or [] for it in items]
    ids = [it.get("id") for it in items]
    return {"text_feats": text_feats, "image_feats": image_feats, "labels": labels, "ids": ids}
