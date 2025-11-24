from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TaskMessage:
    id: str
    from_role: str
    to_role: str
    kind: Literal["plan", "code_suggestion", "test_result", "brain_state", "critique"]
    content: str
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


__all__ = ["TaskMessage"]
