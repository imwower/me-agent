from __future__ import annotations

from typing import Dict, List

from .experiment_types import ExperimentScenario


class ExperimentScenarioRegistry:
    def __init__(self) -> None:
        self._scenarios: Dict[str, ExperimentScenario] = {}

    def register(self, scenario: ExperimentScenario) -> None:
        self._scenarios[scenario.id] = scenario

    def get(self, scenario_id: str) -> ExperimentScenario | None:
        return self._scenarios.get(scenario_id)

    def list_ids(self) -> List[str]:
        return list(self._scenarios.keys())


__all__ = ["ExperimentScenarioRegistry"]
