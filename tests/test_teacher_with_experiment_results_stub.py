from __future__ import annotations

import unittest

from me_core.teachers import DummyTeacher
from me_core.teachers.types import TeacherInput, TeacherOutput
from me_core.tasks.experiment_types import ExperimentResult, ExperimentStep
from me_core.memory import Episode
from me_core.introspection import IntrospectionLog


class TeacherExperimentTestCase(unittest.TestCase):
    def test_dummy_teacher_config_patch(self) -> None:
        teacher = DummyTeacher()
        step = ExperimentStep(repo_id="repo", kind="train", command=["echo"])
        res = ExperimentResult(step=step, returncode=0, stdout="loss=1.0", stderr="", metrics={"loss": 1.0})
        ti = TeacherInput(
            scenario_id="exp",
            episodes=[],
            introspection=None,
            current_config={},
            notes="experiment",
            experiment_results=[res],
        )
        out: TeacherOutput = teacher.generate_advice(ti)
        self.assertTrue(out.config_patches or out.policy_patches or out.advice_text)


if __name__ == "__main__":
    unittest.main()
