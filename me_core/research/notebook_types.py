from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal


@dataclass
class ExperimentEntry:
    id: str
    timestamp: float
    kind: Literal["benchmark", "devloop", "coevo", "brain_exp"]
    description: str
    config_summary: Dict[str, Any]
    metrics: Dict[str, float]
    notes: str


@dataclass
class ExperimentNotebook:
    id: str
    title: str
    created_at: float = field(default_factory=time.time)
    entries: List[ExperimentEntry] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


__all__ = ["ExperimentEntry", "ExperimentNotebook"]
