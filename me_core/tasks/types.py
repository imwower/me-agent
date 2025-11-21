from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskStep:
    user_input: str
    image_path: Optional[str] = None
    expected_keywords: Optional[List[str]] = None
    weight: float = 1.0


@dataclass
class TaskResult:
    success: bool
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    steps: List[TaskStep]
    eval_config: Dict[str, Any] = field(default_factory=dict)
