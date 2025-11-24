from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class LineSeries:
    label: str
    x: List[float]
    y: List[float]


@dataclass
class BarSeries:
    label: str
    categories: List[str]
    values: List[float]


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: Optional[float] = None


@dataclass
class PlotSpec:
    """
    通用图表规格，供 me_ext 后端渲染为实际图像。
    """

    id: str
    kind: Literal["line", "bar", "brain_graph"]
    title: str
    x_label: str = ""
    y_label: str = ""
    line_series: List[LineSeries] = field(default_factory=list)
    bar_series: List[BarSeries] = field(default_factory=list)
    graph_nodes: List[str] = field(default_factory=list)
    graph_edges: List[GraphEdge] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


__all__ = ["LineSeries", "BarSeries", "GraphEdge", "PlotSpec"]
