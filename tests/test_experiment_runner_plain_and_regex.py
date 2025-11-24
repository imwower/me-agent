from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

from me_core.tasks import ExperimentScenario, ExperimentStep, run_experiment_scenario, evaluate_experiment_results
from me_core.workspace import RepoSpec, Workspace


class ExperimentRunnerTestCase(unittest.TestCase):
    def test_regex_and_json_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            spec = RepoSpec(id="repo", name="repo", path=str(repo_path), allowed_paths=["."])
            ws = Workspace([spec])

            steps = [
                ExperimentStep(
                    repo_id="repo",
                    kind="train",
                    command=[sys.executable, "-c", "print('loss=0.42')"],
                    parse_mode="regex",
                    parse_pattern=r"loss=([0-9\\.]+)",
                    metrics_keys=["loss"],
                ),
                ExperimentStep(
                    repo_id="repo",
                    kind="eval",
                    command=[sys.executable, "-c", f"import json; print(json.dumps({{'acc': 0.9}}))"],
                    parse_mode="json",
                    metrics_keys=["acc"],
                ),
            ]
            scenario = ExperimentScenario(
                id="exp1",
                name="demo",
                description="regex+json",
                steps=steps,
                eval_formula="1 - train_loss + eval_acc",
            )
            results = run_experiment_scenario(ws, scenario)
            self.assertEqual(len(results), 2)
            self.assertAlmostEqual(results[0].metrics.get("loss", 0.0), 0.42, places=2)
            self.assertAlmostEqual(results[1].metrics.get("acc", 0.0), 0.9, places=2)
            score = evaluate_experiment_results(results, scenario.eval_formula)
            self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
