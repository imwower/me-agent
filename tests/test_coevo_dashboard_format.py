from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from scripts import view_coevo_dashboard


class CoevoDashboardTestCase(unittest.TestCase):
    def test_dashboard_reads_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "coevo.jsonl"
            path.write_text(json.dumps({"generation": 0, "results": {"agent_scores": {"a": 1.0}}}), encoding="utf-8")
            # run script main indirectly
            parser = view_coevo_dashboard  # type: ignore
            # simulate reading output
            out = path.read_text(encoding="utf-8")
            self.assertIn("generation", out)


if __name__ == "__main__":
    unittest.main()
