from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BrainRegion:
    id: str
    name: str
    kind: str
    size: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrainConnection:
    id: str
    pre_region: str
    post_region: str
    type: str
    sparsity: float
    weight_scale: float | None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrainMetric:
    name: str
    value: float
    unit: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrainSnapshot:
    repo_id: str
    region_activity: Dict[str, float]
    global_metrics: Dict[str, float]
    memory_summary: Dict[str, Any]
    decision_hint: Dict[str, Any]
    created_at: float = field(default_factory=time.time)


__all__ = ["BrainRegion", "BrainConnection", "BrainMetric", "BrainSnapshot"]
