#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _download_split(root: Path, split: str) -> None:
    """下载并落盘指定 split 的 Chinese-SimpleVQA 数据，并输出写入进度。"""

    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("开始加载 Chinese-SimpleVQA: split=%s, cache_dir=%s", split, cache_dir)
    ds = load_dataset(
        "OpenStellarTeam/Chinese-SimpleVQA",
        split=split,
        cache_dir=str(cache_dir),
    )

    total = len(ds)
    out_path = root / f"chinese_simplevqa_{split}.jsonl"
    logger.info("开始写入本地文件: %s（样本数=%d）", out_path, total)

    with out_path.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")

            # 每处理固定步数输出一次进度
            if (idx + 1) % 1000 == 0 or (idx + 1) == total:
                percent = (idx + 1) * 100.0 / total if total > 0 else 0.0
                logger.info(
                    "split=%s: 已写入 %d/%d 条样本（%.1f%%）",
                    split,
                    idx + 1,
                    total,
                    percent,
                )

    logger.info("完成写入: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/vqa_cn",
        help="数据根目录，将在其中创建 cache/ 与 jsonl 文件",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=["train"],
        help='需要下载的切分名称列表，例如: --splits train validation test，默认仅 "train"',
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    split_names: List[str] = args.splits
    logger.info("目标根目录: %s, 计划下载的 splits: %s", root, ", ".join(split_names))

    for split in split_names:
        try:
            _download_split(root, split)
        except Exception as e:  # noqa: BLE001
            logger.warning("下载或写入 split=%s 失败，将跳过。错误: %s", split, e)


if __name__ == "__main__":
    main()

