from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

from me_core.policy.agents import AgentSpec
from me_core.policy.loader import load_policy_from_file
from me_core.config import load_agent_config
from me_core.tasks import ExperimentScenario, ExperimentStep
from me_core.workspace import RepoSpec, Workspace
from scripts import run_devloop


class DevLoopExperimentModeTestCase(unittest.TestCase):
    def test_run_devloop_with_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            ws = Workspace([RepoSpec(id="repo", name="repo", path=str(repo_path), allowed_paths=["."])])
            agent_spec = AgentSpec(id="dev-agent", config=load_agent_config(None), policy=load_policy_from_file(None))
            output_path = repo_path / "devloop.jsonl"
            exp_step = ExperimentStep(
                repo_id="repo",
                kind="train",
                command=[sys.executable, "-c", 'import json; print(json.dumps({"loss": 0.2}))'],
                parse_mode="json",
                metrics_keys=["loss"],
            )
            exp = ExperimentScenario(
                id="exp_demo",
                name="exp_demo",
                description="demo",
                steps=[exp_step],
                eval_formula="1 - train_loss",
            )
            summary = run_devloop.run_devloop(
                workspace=ws,
                repo_id="repo",
                scenario_ids=[],
                agent_spec=agent_spec,
                teacher_cfg={},
                codellm_cfg={"mode": "mock", "mock_response": json.dumps({"path": "README.md", "content": "demo"})},
                output=output_path,
                experiment_scenarios=[exp],
                brain_mode=False,
            )
            self.assertTrue(output_path.exists())
            self.assertIn("experiments", summary)
            self.assertTrue(summary["experiments"])


if __name__ == "__main__":
    unittest.main()
