from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

from me_core.tools import DumpBrainGraphTool, EvalBrainEnergyTool, EvalBrainMemoryTool
from me_core.workspace import RepoSpec, Workspace


class BrainToolsTestCase(unittest.TestCase):
    def test_braintools_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            structure_out = {
                "regions": [{"id": "r1", "name": "sensory", "kind": "sensory", "size": 10}],
                "connections": [],
                "metrics": [{"name": "energy", "value": 0.1, "unit": "mJ"}],
            }
            (repo_path / "structure.py").write_text(
                f"import json; print(json.dumps({json.dumps(structure_out)}))", encoding="utf-8"
            )
            (repo_path / "energy.py").write_text('import json; print(json.dumps({"energy":0.2,"unit":"mJ"}))', encoding="utf-8")
            (repo_path / "memory.py").write_text('import json; print(json.dumps({"capacity":0.3,"unit":"bits"}))', encoding="utf-8")
            spec = RepoSpec(
                id="repo",
                name="repo",
                path=str(repo_path),
                allowed_paths=["."],
                tags={"brain"},
                meta={
                    "structure_script": [sys.executable, "structure.py"],
                    "energy_script": [sys.executable, "energy.py"],
                    "memory_script": [sys.executable, "memory.py"],
                },
            )
            ws = Workspace([spec])
            gtool = DumpBrainGraphTool(ws)
            etool = EvalBrainEnergyTool(ws)
            mtool = EvalBrainMemoryTool(ws)
            g_res = gtool.run({"repo_id": "repo"})
            self.assertIn("summary", g_res)
            self.assertTrue(g_res.get("metrics"))
            e_res = etool.run({"repo_id": "repo"})
            self.assertAlmostEqual(e_res.get("energy"), 0.2)
            m_res = mtool.run({"repo_id": "repo"})
            self.assertAlmostEqual(m_res.get("capacity"), 0.3)


if __name__ == "__main__":
    unittest.main()
