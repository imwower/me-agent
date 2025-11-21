from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from me_core.policy.agents import AgentSpec


@dataclass
class AgentPopulation:
    specs: List[AgentSpec] = field(default_factory=list)

    def register(self, spec: AgentSpec) -> None:
        self.specs.append(spec)

    def get_specs(self) -> List[AgentSpec]:
        return list(self.specs)


__all__ = ["AgentPopulation"]
