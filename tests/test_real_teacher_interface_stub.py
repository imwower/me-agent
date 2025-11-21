from __future__ import annotations

import unittest

from me_core.teachers.types import TeacherInput
from me_ext.teachers.real_teacher import RealTeacher


class RealTeacherInterfaceStubTestCase(unittest.TestCase):
    def test_generate_advice_with_cli_stub(self) -> None:
        teacher = RealTeacher({"mode": "cli"})
        teacher._call_cli_llm = lambda prompt: '{"advice": "ok", "patches": [{"target": "drives", "path": "curiosity.min_concept_count", "value": 2, "reason": "test"}]}'  # type: ignore[assignment]
        ti = TeacherInput(scenario_id="s1", episodes=[], introspection=None, current_config={})
        out = teacher.generate_advice(ti)
        self.assertTrue(out.policy_patches)
        self.assertEqual(out.policy_patches[0].path, "curiosity.min_concept_count")


if __name__ == "__main__":
    unittest.main()
