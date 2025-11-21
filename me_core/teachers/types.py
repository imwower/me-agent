from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from me_core.memory import Episode
from me_core.introspection import IntrospectionLog


@dataclass
class TeacherInput:
    scenario_id: Optional[str]
    episodes: List[Episode]
    introspection: Optional[IntrospectionLog]
    current_config: Dict[str, Any]
    notes: Optional[str] = None


@dataclass
class PolicyPatch:
    target: str
    path: str
    value: Any
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"target": self.target, "path": self.path, "value": self.value, "reason": self.reason}


@dataclass
class TeacherOutput:
    advice_text: str
    policy_patches: List[PolicyPatch] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
