from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ConfigPoint:
    id: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    notes: str = ""


__all__ = ["ConfigPoint"]
