from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from me_core.workspace.discovery import scan_local_repo_for_tools, generate_workspace_entry_from_profile


class WorkspaceDiscoveryTestCase(unittest.TestCase):
    def test_scan_local_repo_for_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "train_demo.py").write_text("print('train')", encoding="utf-8")
            (scripts / "eval_demo.py").write_text("print('eval')", encoding="utf-8")
            profile = scan_local_repo_for_tools(str(root))
            self.assertIn("run_train", profile.detected_tools)
            entry = generate_workspace_entry_from_profile(profile)
            self.assertEqual(entry["id"], root.name)
            self.assertIn("default_train_cmd", entry["meta"])


if __name__ == "__main__":
    unittest.main()
