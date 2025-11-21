from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from me_core.policy.agents import AgentSpec
from me_core.population.population import AgentPopulation
from me_core.population.runner import evaluate_population
from me_core.config import AgentConfig
from me_core.policy import AgentPolicy
from me_core.teachers import DummyTeacher, TeacherManager


class EvaluatePopulationTestCase(unittest.TestCase):
    def test_evaluate_population_runs(self) -> None:
        spec = AgentSpec(id="spec1", config=AgentConfig(), policy=AgentPolicy())
        population = AgentPopulation([spec])
        tm = TeacherManager([DummyTeacher()])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.jsonl"
            results = evaluate_population(
                population,
                scenario_ids=["self_intro"],
                teacher_manager=tm,
                generations=1,
                output_path=out,
            )
            self.assertIn("spec1", results)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
