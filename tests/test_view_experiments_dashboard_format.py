from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


class ViewExperimentsDashboardTestCase(unittest.TestCase):
    def test_dashboard_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "report.jsonl"
            sample = {"scenario_id": "s1", "score": 0.8}
            report.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "scripts/view_experiments_dashboard.py", "--report", str(report)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0)
            self.assertIn("s1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
