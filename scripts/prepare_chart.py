#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.datamodules.chart import load_chartqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/chartqa", help="ChartQA 数据根目录")
    parser.add_argument("--out", type=str, default="data/chartqa/parsed/chart_samples.jsonl", help="输出 JSONL 路径")
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="加载数据时的抽样比例",
    )
    args = parser.parse_args()

    root = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "开始将 ChartQA 数据转换为统一 schema: input=%s, out=%s, sample_ratio=%.3f",
        root,
        out_path,
        args.sample_ratio,
    )

    samples = load_chartqa(
        split="train",
        sample_ratio=args.sample_ratio,
        cache_dir=str(root / "cache"),
    )

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for uni in samples:
            data = uni.to_dict()
            # 为保证 id 字段存在，可使用 meta+索引在后续扩展
            data.setdefault("id", f"chart_{count}")
            data.setdefault("task", "chart_qa")
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1

    logger.info("ChartQA 预处理完成，共写入 %d 条样本到 %s", count, out_path)


if __name__ == "__main__":
    main()

