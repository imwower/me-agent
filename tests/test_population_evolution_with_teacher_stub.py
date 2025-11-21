from __future__ import annotations

import unittest

from me_core.config import AgentConfig
from me_core.policy import AgentPolicy
from me_core.policy.agents import AgentSpec
from me_core.population.population import AgentPopulation
from me_core.population.runner import evaluate_population
from me_core.teachers.manager import TeacherManager
from me_core.teachers.interface import Teacher
from me_core.teachers.types import PolicyPatch, TeacherInput, TeacherOutput


class PatchTeacher(Teacher):
    name = "patch_teacher"

    def generate_advice(self, ti: TeacherInput) -> TeacherOutput:
        patch = PolicyPatch(target="drives", path="curiosity.min_concept_count", value=1, reason="test")
        return TeacherOutput(advice_text="adjust curiosity", policy_patches=[patch])


class PopulationEvolutionTeacherStubTestCase(unittest.TestCase):
    def test_policy_updates_with_teacher(self) -> None:
        spec = AgentSpec(id="s1", config=AgentConfig(), policy=AgentPolicy())
        pop = AgentPopulation([spec])
        tm = TeacherManager([PatchTeacher()])
        results = evaluate_population(pop, ["self_intro"], teacher_manager=tm, generations=1, output_path=None)
        self.assertIn("s1", results)
        self.assertEqual(spec.policy.curiosity.min_concept_count, 1)


if __name__ == "__main__":
    unittest.main()
