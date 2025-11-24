from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from me_core.policy.agents import AgentSpec


@dataclass
class AgentFitness:
    spec_id: str
    scenario_scores: Dict[str, float]
    experiment_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    notes: str = ""
    introspection_summaries: List[str] = field(default_factory=list)


__all__ = ["AgentFitness", "AgentSpec"]
