from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from me_core.workspace import RepoSpec, Workspace
from scripts import run_devloop


class DevLoopStubCodeLLMTestCase(unittest.TestCase):
    def test_run_devloop_with_mock_codellm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            spec = RepoSpec(id="repo", name="repo", path=str(repo_path), allowed_paths=["."])
            ws = Workspace([spec])
            codellm_cfg = {
                "mode": "mock",
                "mock_response": json.dumps({"file_changes": [{"path": "README.md", "content": "demo change"}]}),
            }
            agent_spec = run_devloop._make_agent_spec(None, None)
            output_path = repo_path / "devloop_report.jsonl"

            summary = run_devloop.run_devloop(
                workspace=ws,
                repo_id="repo",
                scenario_ids=["self_intro"],
                agent_spec=agent_spec,
                teacher_cfg={},
                codellm_cfg=codellm_cfg,
                output=output_path,
            )

            self.assertTrue((repo_path / "README.md").exists())
            self.assertTrue(output_path.exists())
            self.assertTrue(summary["results"])
            changed_files = [cf for r in summary["results"] for t in r["tasks"] for cf in t.get("changed_files", [])]
            self.assertIn("README.md", changed_files)


if __name__ == "__main__":
    unittest.main()
