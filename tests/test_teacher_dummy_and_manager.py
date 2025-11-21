from __future__ import annotations

import unittest

from me_core.memory import Episode
from me_core.teachers import DummyTeacher, TeacherInput, TeacherManager
from me_core.teachers.types import PolicyPatch


class TeacherDummyTestCase(unittest.TestCase):
    def test_dummy_teacher_outputs_advice(self) -> None:
        teacher = DummyTeacher()
        ti = TeacherInput(
            scenario_id="s1",
            episodes=[],
            introspection=None,
            current_config={"last_score": 0.2},
        )
        out = teacher.generate_advice(ti)
        self.assertTrue(out.advice_text)

    def test_manager_aggregate_patches(self) -> None:
        t1 = DummyTeacher()
        tm = TeacherManager([t1])
        out = tm.gather_advice(
            TeacherInput(
                scenario_id=None,
                episodes=[Episode(id="e", start_step=0, end_step=0, events=[], summary="")],
                introspection=None,
                current_config={},
            )
        )
        patches = tm.aggregate_patches(out)
        self.assertIsInstance(patches, list)


if __name__ == "__main__":
    unittest.main()
