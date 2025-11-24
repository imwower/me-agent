from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ExperimentStep:
    repo_id: str
    kind: Literal["train", "eval", "custom"]
    command: List[str]
    parse_mode: Literal["regex", "json", "plain"] = "plain"
    parse_pattern: Optional[str] = None
    metrics_keys: Optional[List[str]] = None


@dataclass
class ExperimentResult:
    step: ExperimentStep
    returncode: int
    stdout: str
    stderr: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentScenario:
    id: str
    name: str
    description: str
    steps: List[ExperimentStep]
    eval_formula: str = "0.0"  # 例如 "1 - loss" 或 "acc"


__all__ = ["ExperimentStep", "ExperimentResult", "ExperimentScenario"]
