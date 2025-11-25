"""训练轻量多模态 backend（占位实现，支持小数据集对比学习）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from me_ext.multimodal_backend.datasets import (  # noqa: E402
    build_train_eval_splits,
    MultimodalDataset,
    collate_batch,
)
from me_ext.multimodal_backend.models import MultimodalBackbone  # noqa: E402
from me_ext.multimodal_backend.trainer import MultimodalTrainer  # noqa: E402


def run_training(
    internal_data: List[str],
    external_data: List[str],
    model_name: str,
    output_dir: str,
    freeze_backbone: bool,
    batch_size: int,
    max_steps: int,
    device: str,
) -> None:
    train_examples, eval_examples = build_train_eval_splits(internal_data, external_data, eval_ratio=0.1, max_samples=5000)
    if not train_examples:
        raise SystemExit("未找到任何多模态训练样本")
    train_ds = MultimodalDataset(train_examples)
    eval_ds = MultimodalDataset(eval_examples) if eval_examples else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch) if eval_ds else None

    model = MultimodalBackbone(input_dim=train_ds.feature_dim, proj_dim=128, freeze_backbone=freeze_backbone)
    trainer = MultimodalTrainer(model, train_loader, eval_loader, device=device)
    trainer.train_clip_style(max_steps=max_steps, batch_size=batch_size, output_dir=output_dir)
    print(f"训练完成，权重保存到 {output_dir}")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description="训练多模态 backend（轻量对比学习）")
    parser.add_argument("--workspace", type=str, default=None, help="workspace 配置（当前未使用，占位）")
    parser.add_argument("--model-name", type=str, default="stub-multimodal", help="预训练模型名称（占位）")
    parser.add_argument("--internal-data", type=str, required=True, help="内部数据路径（jsonl，逗号分隔可多个）")
    parser.add_argument("--external-data", type=str, default="", help="外部数据路径（jsonl，逗号分隔）")
    parser.add_argument("--freeze-backbone", type=str, default="true", help="是否冻结 backbone（占位）")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="checkpoints/multimodal_backend/stub")
    args = parser.parse_args()

    internal_paths = [p for p in args.internal_data.split(",") if p]
    external_paths = [p for p in args.external_data.split(",") if p]
    freeze_backbone = str(args.freeze_backbone).lower() in {"1", "true", "yes", "y"}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_training(
        internal_data=internal_paths,
        external_data=external_paths,
        model_name=args.model_name,
        output_dir=args.output_dir,
        freeze_backbone=freeze_backbone,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        device=args.device,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
