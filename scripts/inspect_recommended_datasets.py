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
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


DEFAULT_CHECK: List[str] = [
    "swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT",
]


def _auto_find_jsonl(root: Path) -> Optional[Path]:
    files = list(root.rglob("*.jsonl"))
    if not files:
        return None
    return sorted(files, key=lambda p: len(str(p)))[0]


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
        "--root",
        type=Path,
        default=Path("data/modelscope"),
        help="数据集缓存根目录，默认 data/modelscope",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="直接指定 jsonl 文件路径（包含 messages 字段）。",
    )
    args = parser.parse_args()

    root = args.root.expanduser()

    data_path: Optional[Path] = None
    if args.file is not None:
        data_path = args.file.expanduser()
    else:
        data_path = _auto_find_jsonl(root)

    if data_path is None or not data_path.exists():
        logger.error("未找到 jsonl 文件，请用 --file 指定，或确认 data/modelscope 下存在下载文件。")
        return

    logger.info("加载本地 jsonl: %s", data_path)
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("json", data_files={"train": str(data_path)}, split="train")
    _print_samples(ds)

    logger.info("检查完成。")


if __name__ == "__main__":
    main()
