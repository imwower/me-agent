from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from me_core.policy.agents import AgentSpec


@dataclass
class AgentFitness:
    spec_id: str
    scenario_scores: Dict[str, float]
    overall_score: float
    notes: str = ""
    introspection_summaries: List[str] = field(default_factory=list)


__all__ = ["AgentFitness", "AgentSpec"]
