from __future__ import annotations

import json
from typing import Any, Dict

from .graph import BrainGraph
from .types import BrainConnection, BrainMetric, BrainRegion


def parse_brain_graph_from_json(repo_id: str, json_str: str) -> BrainGraph:
    """
    将 self-snn 导出的脑结构 JSON 转为 BrainGraph。
    期望结构：
    {
      "regions": [{"id": "...", "name": "...", "kind": "...", "size": 123, "meta": {...}}, ...],
      "connections": [{"id": "...", "pre_region": "...", "post_region": "...", "type": "...", "sparsity": 0.1, "weight_scale": 1.0, "meta": {...}}, ...],
      "metrics": [{"name": "energy", "value": 1.2, "unit": "mJ", "meta": {...}}, ...],
      "meta": {...}
    }
    """

    obj: Dict[str, Any] = json.loads(json_str)
    graph = BrainGraph(repo_id=repo_id)
    for r in obj.get("regions", []) or []:
        graph.add_region(
            BrainRegion(
                id=str(r.get("id") or r.get("name") or f"region_{len(graph.regions)}"),
                name=str(r.get("name") or r.get("id") or ""),
                kind=str(r.get("kind") or "unknown"),
                size=int(r.get("size") or 0),
                meta=dict(r.get("meta", {}) or {}),
            )
        )
    for c in obj.get("connections", []) or []:
        graph.add_connection(
            BrainConnection(
                id=str(c.get("id") or f"conn_{len(graph.connections)}"),
                pre_region=str(c.get("pre_region") or ""),
                post_region=str(c.get("post_region") or ""),
                type=str(c.get("type") or "unknown"),
                sparsity=float(c.get("sparsity") or 0.0),
                weight_scale=c.get("weight_scale"),
                meta=dict(c.get("meta", {}) or {}),
            )
        )
    for m in obj.get("metrics", []) or []:
        try:
            value = float(m.get("value"))
        except Exception:
            value = 0.0
        graph.add_metric(
            BrainMetric(
                name=str(m.get("name") or "metric"),
                value=value,
                unit=str(m.get("unit") or ""),
                meta=dict(m.get("meta", {}) or {}),
            )
        )
    if isinstance(obj.get("meta"), dict):
        graph.meta.update(obj["meta"])
    return graph


__all__ = ["parse_brain_graph_from_json"]
