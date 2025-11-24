from __future__ import annotations

import unittest

from me_core.tasks.generated.types import TaskTemplate
from me_core.tasks.generated.generator import TaskGenerator


class TaskGeneratorTestCase(unittest.TestCase):
    def test_generate_tasks_from_gaps(self) -> None:
        templates = [
            TaskTemplate(id="mm", kind="multimodal", description="mm", input_schema={}, output_schema={}, difficulty=1),
            TaskTemplate(id="code", kind="codefix", description="code", input_schema={}, output_schema={}, difficulty=2),
        ]
        gen = TaskGenerator(templates)
        introspections = [{"mistakes": ["code"]}]
        benchmark = [{"id": "b1", "kind": "multimodal", "score": 0.4}]
        tasks = gen.generate_tasks_from_gaps(introspections, benchmark, None, max_new_tasks=3)
        self.assertEqual(len(tasks), 3)
        kinds = {t.kind for t in tasks}
        self.assertTrue(kinds)


if __name__ == "__main__":
    unittest.main()
