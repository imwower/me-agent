from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

from me_core.tools import BrainInferTool
from me_core.workspace import RepoSpec, Workspace


class BrainInferToolStubTestCase(unittest.TestCase):
    def test_brain_infer_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            infer_out = {
                "region_activity": {"mcc": 0.2},
                "global_metrics": {"branching_kappa": 1.0},
                "memory_summary": {"n_keys": 1},
                "decision_hint": {"mode": "explore", "confidence": 0.7},
            }
            (repo_path / "brain_infer.py").write_text(
                f"import json; print(json.dumps({json.dumps(infer_out)}))",
                encoding="utf-8",
            )
            spec = RepoSpec(
                id="brain",
                name="brain",
                path=str(repo_path),
                allowed_paths=["."],
                tags={"brain"},
                meta={"brain_infer_script": [sys.executable, "brain_infer.py"]},
            )
            ws = Workspace([spec])
            tool = BrainInferTool(ws)
            res = tool.run({"repo_id": "brain", "task_id": "t1", "text": "stub"})
            snapshot = res.get("snapshot")
            self.assertIsInstance(snapshot, dict)
            self.assertIn("region_activity", snapshot)
            self.assertEqual(snapshot.get("decision_hint", {}).get("mode"), "explore")


if __name__ == "__main__":
    unittest.main()
