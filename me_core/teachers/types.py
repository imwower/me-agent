from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import logging
import time

from me_core.memory import Episode
from me_core.tasks.experiment_types import ExperimentResult
from me_core.introspection import IntrospectionLog
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from me_core.brain.graph import BrainGraph


@dataclass
class TeacherInput:
    scenario_id: Optional[str]
    episodes: List[Episode]
    introspection: Optional[IntrospectionLog]
    current_config: Dict[str, Any]
    notes: Optional[str] = None
    experiment_results: Optional[List[ExperimentResult]] = None
    brain_graph: Optional["BrainGraph"] = None
    brain_snapshot: Any | None = None


@dataclass
class PolicyPatch:
    target: str
    path: str
    value: Any
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "path": self.path, "value": self.value, "reason": self.reason}


@dataclass
class ConfigPatch:
    repo_id: str
    config_path: str
    path: str
    value: Any
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "config_path": self.config_path,
            "path": self.path,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass
class TeacherOutput:
    advice_text: str
    policy_patches: List[PolicyPatch] = field(default_factory=list)
    config_patches: List[ConfigPatch] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    source_teacher_name: str = ""


TEACHER_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "advice_text": {"type": "string"},
        "policy_patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "path": {"type": "string"},
                    "value": {},
                    "reason": {"type": "string"},
                },
                "required": ["target", "path", "value"],
            },
        },
        "config_patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "repo_id": {"type": "string"},
                    "config_path": {"type": "string"},
                    "path": {"type": "string"},
                    "value": {},
                    "reason": {"type": "string"},
                },
                "required": ["repo_id", "config_path", "path", "value"],
            },
        },
    },
    "required": ["advice_text"],
}


def validate_teacher_output(data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    轻量校验 TeacherOutput 结构，避免依赖 jsonschema。
    返回 (是否通过, 错误列表)。
    """

    errors: List[str] = []
    if not isinstance(data, dict):
        return False, ["output is not a dict"]
    if "advice_text" not in data or not isinstance(data["advice_text"], str):
        errors.append("advice_text missing or not string")
    for key, required in [("policy_patches", ["target", "path", "value"]), ("config_patches", ["path", "value"])]:
        if key not in data:
            continue
        if not isinstance(data[key], list):
            errors.append(f"{key} not list")
            continue
        for item in data[key]:
            if not isinstance(item, dict):
                errors.append(f"{key} item not dict")
                continue
            for r in required:
                if r not in item:
                    errors.append(f"{key} item missing {r}")
    return len(errors) == 0, errors
