from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Job:
    id: str
    name: str
    kind: Literal["devloop", "experiment", "population", "benchmark"]
    config: dict[str, Any] = field(default_factory=dict)
    schedule: str = "daily"
    enabled: bool = True


__all__ = ["Job"]
