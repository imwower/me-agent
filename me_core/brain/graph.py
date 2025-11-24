from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .types import BrainConnection, BrainMetric, BrainRegion


@dataclass
class BrainGraph:
    repo_id: str
    regions: Dict[str, BrainRegion] = field(default_factory=dict)
    connections: Dict[str, BrainConnection] = field(default_factory=dict)
    metrics: List[BrainMetric] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)

    def add_region(self, region: BrainRegion) -> None:
        self.regions[region.id] = region

    def add_connection(self, conn: BrainConnection) -> None:
        self.connections[conn.id] = conn

    def add_metric(self, metric: BrainMetric) -> None:
        self.metrics.append(metric)

    def summary(self) -> str:
        region_count = len(self.regions)
        conn_count = len(self.connections)
        metric_parts = [f"{m.name}={m.value}{m.unit}" for m in self.metrics[:3]]
        metrics_text = "；".join(metric_parts) if metric_parts else "无关键指标"
        return f"脑图谱[{self.repo_id}]: 区域 {region_count} 个，连接 {conn_count} 条，指标：{metrics_text}。"


__all__ = ["BrainGraph"]
