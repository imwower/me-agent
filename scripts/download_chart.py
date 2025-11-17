#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/chartqa", help="图表问答数据根目录")
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="抽样比例（0,1]，用于减少本地数据量",
    )
    args = parser.parse_args()

    root = Path(args.root)
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "尝试从 HuggingFace 下载 ChartQA 数据集索引: root=%s, sample_ratio=%.3f",
        root,
        args.sample_ratio,
    )

    try:
        ds = load_dataset(
            "chartqa",
            split="train",
            streaming=True,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "加载 ChartQA 数据集失败: %s\n"
            "请参考官方项目获取数据，并将其放置到 %s。",
            exc,
            root,
        )
        return

    out_path = root / "chartqa_index.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if args.sample_ratio < 1.0:
                if (i % int(1.0 / args.sample_ratio)) != 0:
                    continue
            item = {
                "id": ex.get("id", f"chart_{i}"),
                "question": ex.get("question"),
                "answer": ex.get("answer"),
            }
            f.write(f"{item}\n")

    logger.info("ChartQA 索引已写入: %s", out_path)


if __name__ == "__main__":
    main()

