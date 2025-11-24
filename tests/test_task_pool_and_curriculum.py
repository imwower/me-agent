from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from me_core.tasks.generated.pool import TaskPool
from me_core.tasks.generated.curriculum import CurriculumPolicy, CurriculumScheduler
from me_core.memory.log_index import LogIndex


class TaskPoolCurriculumTestCase(unittest.TestCase):
    def test_task_pool_and_curriculum(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            task = {
                "id": "t1",
                "template_id": "tpl",
                "payload": {},
                "expected_behavior": "",
                "labels": {},
                "difficulty": 1,
                "kind": "multimodal",
            }
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            pool = TaskPool(tmpdir)
            tasks = pool.list_tasks()
            self.assertEqual(len(tasks), 1)

            log = root / "exp.jsonl"
            log.write_text(json.dumps({"kind": "experiment", "score": 0.5}), encoding="utf-8")
            scheduler = CurriculumScheduler(pool, LogIndex(tmpdir))
            selected = scheduler.select_next_tasks(CurriculumPolicy(mode="easy2hard", max_per_round=1), [])
            self.assertEqual(len(selected), 1)


if __name__ == "__main__":
    unittest.main()
