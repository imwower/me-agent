from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.datamodules.base import OcrToken, UnifiedSample

logger = logging.getLogger(__name__)


def _load_rects_index(root: str, split: str) -> List[Dict[str, Any]]:
    """从本地 ReCTS 目录加载索引信息。

    约定目录结构（示例）：
        root/
          train/
            images/
            annotations.jsonl

    说明：
        - ReCTS 数据集需要注册获取，脚本不会尝试自动下载；
        - 若未找到 annotations.jsonl，则打印如何放置数据的说明，
          并返回空列表。
    """

    base = Path(root) / split
    ann_path = base / "annotations.jsonl"
    if not ann_path.exists():
        logger.warning(
            "未找到 ReCTS 标注文件: %s\n"
            "请在 data/ocr_scene/rects 下创建类似结构：\n"
            "  %s\n"
            "其中 annotations.jsonl 每行含有至少 image/texts 字段。\n"
            "ReCTS 官方信息可参考:\n"
            "  https://rrc.cvc.uab.es/?ch=12\n",
            ann_path,
            ann_path,
        )
        return []

    items: List[Dict[str, Any]] = []
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def load_rects(
    root: str,
    split: str = "train",
    sample_ratio: float = 1.0,
) -> Iterable[UnifiedSample]:
    """加载本地 ReCTS 子集并转换为统一 schema。

    每条样本结构示意（annotations.jsonl 中的单行）：
        {
          "id": "xxxx",
          "image": "images/xxx.jpg",
          "texts": [
            {"text": "...", "bbox": [x1,y1,x2,y2], "conf": 0.99},
            ...
          ]
        }

    若实际字段不同，用户可参考本函数对其进行调整。
    """

    items = _load_rects_index(root, split)
    if not items:
        return []

    base = Path(root) / split

    for idx, item in enumerate(items):
        if sample_ratio < 1.0:
            if (idx % int(1.0 / sample_ratio)) != 0:
                continue

        image_rel = item.get("image") or item.get("image_path")
        image_path = (
            str((base / image_rel).resolve()) if image_rel is not None else ""
        )

        texts = item.get("texts") or []
        ocr_tokens: List[Dict[str, Any]] = []
        for t_idx, tok in enumerate(texts):
            text = str(tok.get("text") or "")
            bbox = tok.get("bbox") or [0, 0, 0, 0]
            conf = float(tok.get("conf", 0.9))
            ocr_tokens.append(
                OcrToken(
                    id=f"t{t_idx}",
                    text=text,
                    bbox=[int(v) for v in bbox],
                ).__dict__
                | {"conf": conf}
            )

        answers: List[str] = []
        if ocr_tokens:
            # 简单地将全部文本拼接为一个“读取全部文字”的答案
            answers = ["".join(t["text"] for t in ocr_tokens)]

        unified = UnifiedSample(
            image=image_path,
            question="请读出图像中的主要文字。",
            answers=answers,
            answerable=True if answers else None,
            evidence={
                "ocr_tokens": ocr_tokens,
                "regions": [],
                "chart_elements": [],
                "table_cells": [],
            },
            meta={
                "dataset": "ReCTS",
                "split": split,
            },
        )
        yield unified

