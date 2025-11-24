from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

from me_core.config import load_agent_config
from me_core.policy import load_policy_from_file
from me_core.policy.agents import AgentSpec
from me_core.population import AgentPopulation
from me_core.population.runner import evaluate_population
from me_core.tasks import ExperimentScenario, ExperimentStep
from me_core.workspace import RepoSpec, Workspace


class PopulationExperimentFitnessTestCase(unittest.TestCase):
    def test_fitness_with_experiment_scores(self) -> None:
        spec = AgentSpec(id="a1", config=load_agent_config(None), policy=load_policy_from_file(None))
        pop = AgentPopulation([spec])
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            ws = Workspace([RepoSpec(id="repo", name="repo", path=str(repo_path), allowed_paths=["."])])
            exp_step = ExperimentStep(
                repo_id="repo",
                kind="eval",
                command=[sys.executable, "-c", 'import json; print(json.dumps({"acc": 0.8}))'],
                parse_mode="json",
                metrics_keys=["acc"],
            )
            exp_sc = ExperimentScenario(
                id="exp",
                name="exp",
                description="demo",
                steps=[exp_step],
                eval_formula="eval_acc",
            )
            fitness = evaluate_population(
                pop,
                scenario_ids=[],
                teacher_manager=None,
                experiment_scenarios=[exp_sc],
                workspace=ws,
                experiment_weight=0.5,
            )
            self.assertIn("a1", fitness)
            self.assertTrue(fitness["a1"].experiment_scores)


if __name__ == "__main__":
    unittest.main()
