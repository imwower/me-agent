from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_chinese_simplevqa(
    split: str = "train",
    sample_ratio: float = 0.1,
    cache_dir: str = "data/cache",
    output_path: Optional[str] = None,
) -> None:
    """下载并抽样 Chinese-SimpleVQA，保存为统一 JSONL 格式。

    每条样本的结构与 src/datamodules/base.UnifiedSample.to_dict 一致。
    """

    from src.datamodules.chinese_simplevqa import convert_example

    ds = load_dataset(
        "OpenStellarTeam/Chinese-SimpleVQA",
        split=split,
        streaming=True,
        cache_dir=cache_dir,
    )

    output = Path(output_path or f"data/chinese_simplevqa_{split}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "开始准备 Chinese-SimpleVQA: split=%s, sample_ratio=%.3f, 输出=%s",
        split,
        sample_ratio,
        output,
    )

    with output.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if sample_ratio < 1.0 and (i % int(1.0 / sample_ratio)) != 0:
                continue
            unified = convert_example(ex).to_dict()
            f.write(f"{unified}\n")

    logger.info("完成 Chinese-SimpleVQA 准备，共写入若干抽样样本到 %s", output)


if __name__ == "__main__":
    prepare_chinese_simplevqa()

