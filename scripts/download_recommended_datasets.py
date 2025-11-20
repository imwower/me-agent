#!/usr/bin/env python
from __future__ import annotations

"""从 ModelScope 下载指定数据集（默认 Qwen3 蒸馏 110k SFT 数据）。

说明：
    - 依赖 modelscope：未安装时提示 `pip install modelscope`；
    - 默认下载位置：data/modelscope/<dataset_id>；
    - 默认下载列表可用 --datasets 覆盖（逗号分隔），但不再调用 HubApi.download_dataset，
      统一使用 MsDataset.load 触发下载。

示例：
    python scripts/download_recommended_datasets.py
    python scripts/download_recommended_datasets.py --datasets swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT
"""

import argparse
import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# 默认仅下载基于 Qwen3 的 110k 蒸馏 SFT 数据
DEFAULT_DATASETS: List[str] = [
    "swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT",
]


def _parse_dataset_list(raw: str | None) -> List[str]:
    if not raw:
        return DEFAULT_DATASETS
    return [item.strip() for item in raw.split(",") if item.strip()]


def _download(api, dataset_id: str, target_root: Path, revision: str = "master") -> None:
    """使用 MsDataset.load 触发下载并缓存到目标目录。"""

    logger.info("开始下载: %s", dataset_id)
    try:
        from modelscope.msdatasets import MsDataset  # type: ignore
    except Exception:
        logger.error("未找到 modelscope，请先运行: pip install modelscope")
        return

    try:
        ds = MsDataset.load(
            dataset_id,
            split="train",
            cache_dir=str(target_root),
            revision=revision,
        )
        logger.info("下载完成: %s，样本数=%d", dataset_id, len(ds))
    except Exception as exc:  # noqa: BLE001
        logger.error("下载失败: %s (%s)", dataset_id, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="下载推荐的中文多任务/多模态数据集（ModelScope）")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="自定义数据集列表，逗号分隔；默认使用内置推荐列表。",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/modelscope"),
        help="下载根目录，默认 data/modelscope",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="master",
        help="数据集版本，默认 master",
    )
    args = parser.parse_args()

    dataset_list = _parse_dataset_list(args.datasets)
    target_root = args.target.expanduser()
    target_root.mkdir(parents=True, exist_ok=True)

    for ds in dataset_list:
        _download(None, ds, target_root, revision=args.revision)

    logger.info("全部尝试完成。已下载的数据位于: %s", target_root)


if __name__ == "__main__":
    main()
