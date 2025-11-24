from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
