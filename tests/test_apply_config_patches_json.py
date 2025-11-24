from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from me_core.codetasks import apply_config_patches
from me_core.teachers.types import ConfigPatch
from me_core.workspace import RepoSpec, Workspace


class ConfigPatchTestCase(unittest.TestCase):
    def test_apply_json_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir)
            cfg_file = repo_path / "config.json"
            cfg_file.write_text(json.dumps({"training": {"lr": 1.0}}), encoding="utf-8")
            ws = Workspace([RepoSpec(id="r", name="r", path=str(repo_path), allowed_paths=["."])])
            patch = ConfigPatch(
                repo_id="r",
                config_path="config.json",
                path="training.lr",
                value=0.5,
                reason="test",
            )
            apply_config_patches(ws, [patch])
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            self.assertEqual(data["training"]["lr"], 0.5)


if __name__ == "__main__":
    unittest.main()
