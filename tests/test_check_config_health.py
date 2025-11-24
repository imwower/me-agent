from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


class ConfigHealthTestCase(unittest.TestCase):
    def test_check_config_health_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ws = {
                "repos": [
                    {
                        "id": "brain",
                        "name": "brain",
                        "path": str(root),
                        "allowed_paths": ["."],
                        "tags": ["brain"],
                        "meta": {
                            "structure_script": ["echo"],
                            "energy_script": ["echo"],
                            "memory_script": ["echo"],
                            "default_train_cmd": ["echo"],
                            "default_eval_cmd": ["echo"],
                        },
                    }
                ]
            }
            ws_path = root / "ws.json"
            ws_path.write_text(json.dumps(ws), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "scripts/check_config_health.py", "--workspace", str(ws_path)],
                cwd=Path(__file__).resolve().parents[1],
            )
            self.assertEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
