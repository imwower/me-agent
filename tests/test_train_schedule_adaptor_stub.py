from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from me_core.tasks.train_schedule import TrainSchedule, export_tasks_for_snn
from me_core.tasks.generated.types import GeneratedTask


class TrainScheduleAdaptorTestCase(unittest.TestCase):
    def test_export_tasks_for_snn(self) -> None:
        task = GeneratedTask(
            id="t1",
            template_id="tpl",
            payload={"note": "x"},
            expected_behavior="do it",
            labels={},
            difficulty=1,
            kind="multimodal",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_tasks_for_snn([task], tmpdir)
            self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
