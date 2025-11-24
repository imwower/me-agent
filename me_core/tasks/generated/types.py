from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal


@dataclass
class TaskTemplate:
    id: str
    kind: Literal["multimodal", "codefix", "brain_memory", "brain_control"]
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    difficulty: int = 1


@dataclass
class GeneratedTask:
    id: str
    template_id: str
    payload: Dict[str, Any]
    expected_behavior: str
    labels: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)
    difficulty: int = 1
    kind: str | None = None


__all__ = ["TaskTemplate", "GeneratedTask"]
