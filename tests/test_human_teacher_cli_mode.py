from __future__ import annotations

import json
import tempfile
import unittest

from me_core.teachers.interface import HumanTeacher
from me_core.teachers.types import TeacherInput


class HumanTeacherTestCase(unittest.TestCase):
    def test_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/human.json"
            teacher = HumanTeacher(input_mode="file", file_path=path)
            ti = TeacherInput(scenario_id="s1", episodes=[], introspection=None, current_config={}, notes="hi")
            # 写入文件后模拟用户填写
            teacher.generate_advice(ti)  # prompts and waits for input; we cannot block
            # Instead, directly write expected JSON and call again
            json.dump({"advice_text": "ok", "policy_patches": []}, open(path, "w", encoding="utf-8"))
            out = teacher.generate_advice(ti)
            self.assertIn("ok", out.advice_text)


if __name__ == "__main__":
    unittest.main()
