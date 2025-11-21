from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class IntrospectionLog:
    id: str
    created_at: float
    scenario_id: Optional[str]
    step_range: Tuple[int, int]
    summary: str
    mistakes: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "scenario_id": self.scenario_id,
            "step_range": list(self.step_range),
            "summary": self.summary,
            "mistakes": list(self.mistakes),
            "improvements": list(self.improvements),
        }

    @classmethod
    def new(
        cls,
        scenario_id: Optional[str],
        step_range: Tuple[int, int],
        summary: str,
        mistakes: List[str],
        improvements: List[str],
    ) -> "IntrospectionLog":
        return cls(
            id=str(uuid.uuid4()),
            created_at=time.time(),
            scenario_id=scenario_id,
            step_range=step_range,
            summary=summary,
            mistakes=mistakes,
            improvements=improvements,
        )
