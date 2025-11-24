from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

from me_core.workspace import RepoSpec, Workspace
from scripts import run_devloop


class BrainGuidedDecisionScenarioTestCase(unittest.TestCase):
    def test_brain_guided_decision_with_stub_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target_repo = root / "target"
            brain_repo = root / "brain"
            target_repo.mkdir()
            brain_repo.mkdir()

            structure_out = {
                "regions": [{"id": "mcc", "name": "MCC", "kind": "core", "size": 10}],
                "connections": [],
                "metrics": [{"name": "num_regions", "value": 1, "unit": ""}],
            }
            (brain_repo / "structure.py").write_text(
                f"import json; print(json.dumps({json.dumps(structure_out)}))", encoding="utf-8"
            )
            (brain_repo / "energy.py").write_text(
                'import json; print(json.dumps({"energy":0.1,"unit":"j"}))', encoding="utf-8"
            )
            (brain_repo / "memory.py").write_text(
                'import json; print(json.dumps({"capacity":1.0,"unit":"bits"}))', encoding="utf-8"
            )
            infer_out = {
                "region_activity": {"mcc": 0.2},
                "global_metrics": {"branching_kappa": 1.0},
                "memory_summary": {"n_keys": 1},
                "decision_hint": {"mode": "explore", "confidence": 0.8},
            }
            (brain_repo / "brain_infer.py").write_text(
                f"import json; print(json.dumps({json.dumps(infer_out)}))",
                encoding="utf-8",
            )

            target_spec = RepoSpec(id="target", name="target", path=str(target_repo), allowed_paths=["."])
            brain_spec = RepoSpec(
                id="brain",
                name="brain",
                path=str(brain_repo),
                allowed_paths=["."],
                tags={"brain"},
                meta={
                    "structure_script": [sys.executable, "structure.py"],
                    "energy_script": [sys.executable, "energy.py"],
                    "memory_script": [sys.executable, "memory.py"],
                    "brain_infer_script": [sys.executable, "brain_infer.py"],
                    "default_config": "configs/agency.yaml",
                },
            )
            ws = Workspace([target_spec, brain_spec])

            codellm_cfg = {
                "mode": "mock",
                "mock_response": json.dumps({"file_changes": [{"path": "README.md", "content": "brain-guided"}]}),
            }
            agent_spec = run_devloop._make_agent_spec(None, None)
            output_path = target_repo / "report.jsonl"

            summary = run_devloop.run_devloop(
                workspace=ws,
                repo_id="target",
                scenario_ids=["brain_guided_decision"],
                agent_spec=agent_spec,
                teacher_cfg={},
                codellm_cfg=codellm_cfg,
                output=output_path,
                brain_mode=True,
            )

            self.assertTrue(summary.get("brain_snapshots"))
            snap = summary["brain_snapshots"][-1]
            self.assertEqual(snap.get("decision_hint", {}).get("mode"), "explore")
            self.assertTrue((target_repo / "README.md").exists())


if __name__ == "__main__":
    unittest.main()
