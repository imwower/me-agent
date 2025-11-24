from __future__ import annotations

import unittest

from me_core.agent import SimpleAgent
from me_core.agent.multi_agent import MultiAgentCoordinator
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.learning import SimpleLearner
from me_core.perception import TextPerception
from me_core.self_model import SimpleSelfModel
from me_core.tasks.types import Scenario, TaskStep
from me_core.world_model import SimpleWorldModel


def _build_agent() -> SimpleAgent:
    return SimpleAgent(
        perception=TextPerception(),
        world_model=SimpleWorldModel(),
        self_model=SimpleSelfModel(),
        drive_system=SimpleDriveSystem(),
        tools={},
        learner=SimpleLearner(),
        dialogue_policy=RuleBasedDialoguePolicy(),
    )


class MultiAgentCoordinatorTestCase(unittest.TestCase):
    def test_run_devloop_task(self) -> None:
        agent = _build_agent()
        ma = MultiAgentCoordinator.from_single_agent(agent)
        scenario = Scenario(
            id="s1",
            name="test",
            description="",
            steps=[TaskStep(user_input="hi", expected_keywords=["hi"])],
        )
        res = ma.run_devloop_task(scenario)
        self.assertIn("roles", res)
        self.assertIn("coder", res["roles"])


if __name__ == "__main__":
    unittest.main()
