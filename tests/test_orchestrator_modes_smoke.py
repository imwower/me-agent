from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


class OrchestratorSmokeTestCase(unittest.TestCase):
    def test_benchmark_mode(self) -> None:
        proc = subprocess.run(
            [sys.executable, "scripts/run_orchestrator.py", "--mode", "benchmark"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
        data = json.loads(proc.stdout)
        self.assertIn("benchmark", data)


if __name__ == "__main__":
    unittest.main()
