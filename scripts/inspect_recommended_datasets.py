#!/usr/bin/env python
from __future__ import annotations

"""快速检查已下载的 ModelScope 数据集，打印基本信息与少量样本。

说明：
    - 依赖 modelscope；
    - 默认检查 download_recommended_datasets.py 下载到的 data/modelscope 根目录；
    - 仅打印前若干条样本字段，适合确认路径与可读性。
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


DEFAULT_CHECK: List[str] = [
    "swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT",
]


def _load_dataset(dataset_id: str, target_root: Path, split: str = "train") -> Any:
    try:
        from modelscope.msdatasets import MsDataset  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("未找到 modelscope，请先 pip install modelscope") from exc

    logger.info("加载数据集: %s (split=%s)", dataset_id, split)
    ds = MsDataset.load(
        dataset_id,
        subset_name=None,
        revision="master",
        split=split,
        cache_dir=str(target_root),
    )
    return ds


def _print_samples(ds: Any, max_rows: int = 2) -> None:
    try:
        for i, row in enumerate(ds):
            if i >= max_rows:
                break
            logger.info("样本 #%d: %s", i, _truncate_row(row))
    except Exception as exc:  # noqa: BLE001
        logger.error("遍历样本失败: %s", exc)


def _truncate_row(row: Dict[str, Any], max_len: int = 120) -> Dict[str, Any]:
    def _clip(v: Any) -> Any:
        if isinstance(v, str) and len(v) > max_len:
            return v[:max_len] + "...(truncated)"
        return v

    return {k: _clip(v) for k, v in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="检查已下载的 ModelScope 数据集样本")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="要检查的数据集列表，逗号分隔；默认检查少量代表性数据集。",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/modelscope"),
        help="数据集缓存根目录，默认 data/modelscope",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="要加载的数据集切分，默认 train",
    )
    args = parser.parse_args()

    ds_list = DEFAULT_CHECK if not args.datasets else [s.strip() for s in args.datasets.split(",") if s.strip()]
    root = args.root.expanduser()

    for ds_id in ds_list:
        try:
            ds = _load_dataset(ds_id, root, split=args.split)
            _print_samples(ds)
        except Exception as exc:  # noqa: BLE001
            logger.error("检查失败: %s (%s)", ds_id, exc)

    logger.info("检查完成。")


if __name__ == "__main__":
    main()
