#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/ocr_scene", help="OCR 场景数据根目录")
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.3,
        help="占位参数：实际下载需用户自行注册数据集。",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rects_dir = root / "rects" / "train"
    rects_dir.mkdir(parents=True, exist_ok=True)

    logger.warning(
        "ReCTS / RCTW-17 / LSVT 等中文场景文本数据集需要在各自官方网站注册后获取，"
        "本脚本不会尝试自动下载。\n"
        "请访问相关网站下载数据，并将 ReCTS 数据按如下形式放置：\n"
        "  %s/images  # 存放图像\n"
        "  %s/annotations.jsonl  # 每行包含 image/texts 字段的 JSON，详见 src/datamodules/ocr/rects.py\n"
        "当前 sample_ratio=%.2f 仅用于后续加载时抽样使用。",
        rects_dir,
        rects_dir,
        args.sample_ratio,
    )


if __name__ == "__main__":
    main()

