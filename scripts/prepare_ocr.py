#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.datamodules.ocr import load_rects
from src.inference.evidence import collect_ocr_evidence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/ocr_scene", help="OCR 场景数据根目录")
    parser.add_argument("--out", type=str, default="data/ocr_scene/ocr_tokens.jsonl", help="输出 JSONL 路径")
    parser.add_argument(
        "--engine",
        type=str,
        default="pytesseract",
        help="OCR 引擎（当前仅支持 pytesseract，占位参数）",
    )
    parser.add_argument("--lang", type=str, default="zh", help="语言（占位参数）")
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="加载索引时的抽样比例",
    )
    args = parser.parse_args()

    root = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "开始为 OCR 场景数据生成 ocr_tokens.jsonl: root=%s, out=%s",
        root,
        out_path,
    )

    # 使用本地 ReCTS 索引（若存在），否则该迭代器为空
    unified_samples = list(load_rects(str(root / "rects"), split="train", sample_ratio=args.sample_ratio))

    with out_path.open("w", encoding="utf-8") as f:
        for sample in unified_samples:
            # 若已有标注 tokens，则直接使用；否则运行 OCR 引擎
            ev = sample.evidence
            tokens = ev.get("ocr_tokens") or []
            if not tokens:
                tokens = collect_ocr_evidence(sample.image)
            record = {
                "id": sample.meta.get("id") or None,
                "image": sample.image,
                "ocr_tokens": tokens,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("OCR 预处理完成，结果已写入: %s", out_path)


if __name__ == "__main__":
    main()

