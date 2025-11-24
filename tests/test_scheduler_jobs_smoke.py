from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from me_core.scheduler.runner import load_jobs, should_run
from me_core.scheduler.types import Job


class SchedulerSmokeTestCase(unittest.TestCase):
    def test_load_and_should_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            path.write_text(json.dumps({"jobs": [{"id": "j1", "kind": "devloop", "schedule": "daily"}]}), encoding="utf-8")
            jobs = load_jobs(path)
            self.assertEqual(len(jobs), 1)
            self.assertTrue(should_run(jobs[0].schedule, None, 0))


if __name__ == "__main__":
    unittest.main()
