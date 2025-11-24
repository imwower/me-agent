from __future__ import annotations

import json
import unittest

from me_core.teachers.types import validate_teacher_output, TeacherInput
from me_ext.teachers.real_teacher import RealTeacher


class TeacherOutputSchemaTestCase(unittest.TestCase):
    def test_validate_teacher_output(self) -> None:
        ok, errs = validate_teacher_output({"advice_text": "hi", "policy_patches": [{"target": "drives", "path": "a", "value": 1}]})
        self.assertTrue(ok)
        self.assertFalse(errs)

    def test_real_teacher_parse_invalid(self) -> None:
        teacher = RealTeacher({"mode": "cli", "command": "cat"})
        # monkeypatch cli call
        teacher._call_cli_llm = lambda prompt: '{"advice_text": "ok", "policy_patches": [{"target": "x", "value": 1}]}'  # type: ignore
        ti = TeacherInput(scenario_id="s", episodes=[], introspection=None, current_config={}, notes=None)
        out = teacher.generate_advice(ti)
        self.assertEqual(len(out.policy_patches), 0)
        self.assertEqual(out.source_teacher_name, teacher.name)


if __name__ == "__main__":
    unittest.main()
