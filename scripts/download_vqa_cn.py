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
    parser.add_argument("--root", type=str, default="data/vqa_cn", help="数据根目录")
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.5,
        help="抽样比例（0,1]，用于减少本地数据量",
    )
    args = parser.parse_args()

    root = Path(args.root)
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "开始下载 Chinese-SimpleVQA 子集: root=%s, sample_ratio=%.3f",
        root,
        args.sample_ratio,
    )

    ds = load_dataset(
        "OpenStellarTeam/Chinese-SimpleVQA",
        split="train",
        streaming=True,
        cache_dir=str(cache_dir),
    )

    out_path = root / "chinese_simplevqa_index.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if args.sample_ratio < 1.0:
                if (i % int(1.0 / args.sample_ratio)) != 0:
                    continue
            item = {
                "id": ex.get("id", f"cs_{i}"),
                "question": ex.get("final_question") or ex.get("question"),
                "answer": ex.get("final_answer") or ex.get("answer"),
            }
            f.write(f"{item}\n")

    logger.info("Chinese-SimpleVQA 索引已写入: %s", out_path)


if __name__ == "__main__":
    main()

