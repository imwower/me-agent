from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.datamodules.base import ChartElement


def parse_chart_metadata(meta_path: str) -> List[Dict]:
    """从图表元数据文件中解析 chart_elements 列表。

    该函数主要作为 ChartY-zh/ChartQA 等数据集的公共入口，
    具体字段格式由各数据集的 datamodule 再做一次轻量适配。
    """

    path = Path(meta_path)
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    elements: List[Dict] = []
    data_points = meta.get("data", [])
    chart_type = str(meta.get("chart_type") or "unknown")

    for idx, item in enumerate(data_points):
        elem = ChartElement(
            id=f"{chart_type}_{idx}",
            type=chart_type,
            meta=item,
        )
        elements.append(elem.__dict__)

    return elements

