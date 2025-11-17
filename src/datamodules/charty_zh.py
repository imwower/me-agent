from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from .base import ChartElement, UnifiedSample

logger = logging.getLogger(__name__)


def _load_charty_zh_item(meta_path: Path) -> Dict:
    """从 ChartY-zh 的单个样本元数据文件中加载结构。

    约定：
        - meta_path 为 JSON 文件，包含图表类型、数据等信息；
        - 对于实际数据集，用户可根据官方格式调整本函数。
    """

    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_chart_elements(meta: Dict) -> List[Dict]:
    """从图表元数据构造统一的 chart_elements 列表。

    该实现假设 meta 中包含 series/x/y 等字段，仅作为示例占位。
    """

    elements: List[Dict] = []
    data_points = meta.get("data", [])
    for idx, item in enumerate(data_points):
        elem = ChartElement(
            id=f"pt_{idx}",
            type=str(meta.get("chart_type") or "unknown"),
            meta={
                "series": item.get("series"),
                "x": item.get("x"),
                "y": item.get("y"),
            },
        )
        elements.append(elem.__dict__)
    return elements


def load_charty_zh_local(
    root_dir: str,
    split: str = "train",
    sample_ratio: float = 1.0,
) -> Iterable[UnifiedSample]:
    """从本地 ChartY-zh 目录加载样本并转换为统一 schema。

    由于公开数据格式可能存在差异，此处采用占位实现：
        - 假设 root_dir 下存在若干子目录/文件，每个样本包含：
            * chart.png 或 .jpg 图像文件；
            * meta.json 描述图表结构与 QA（若存在）。
        - 若不存在 QA，则构造结构抽取任务（问题为“这个图表的主要数据点有哪些？”）。
    """

    root = Path(root_dir)
    if not root.exists():
        logger.warning("ChartY-zh 本地目录不存在: %s", root_dir)
        return

    meta_files = sorted(root.glob(f"{split}/**/meta.json"))
    logger.info(
        "加载 ChartY-zh 本地样本: root=%s, split=%s, 样本数(文件级)=%d",
        root_dir,
        split,
        len(meta_files),
    )

    for idx, meta_path in enumerate(meta_files):
        if sample_ratio < 1.0:
            if (idx % int(1.0 / sample_ratio)) != 0:
                continue

        meta = _load_charty_zh_item(meta_path)
        image_path = meta_path.with_name("chart.png")
        if not image_path.exists():
            # 尝试 jpg 备选
            jpg_path = meta_path.with_name("chart.jpg")
            image_path = jpg_path if jpg_path.exists() else image_path

        chart_elements = _build_chart_elements(meta)

        qa_list = meta.get("qa", [])
        if qa_list:
            q = str(qa_list[0].get("question") or "这个图表表达了什么？")
            ans = str(qa_list[0].get("answer") or "")
            answers = [ans] if ans else []
        else:
            q = "这个图表的主要数据点有哪些？"
            answers = []

        unified = UnifiedSample(
            image=str(image_path),
            question=q,
            answers=answers,
            answerable=True if answers else None,
            evidence={
                "ocr_tokens": [],
                "regions": [],
                "chart_elements": chart_elements,
            },
            meta={
                "dataset": "charty_zh",
                "split": split,
            },
        )
        yield unified

