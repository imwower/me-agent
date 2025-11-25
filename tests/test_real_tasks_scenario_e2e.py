import unittest

from me_core.agent import SimpleAgent
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.event_stream import EventStream
from me_core.learning import SimpleLearner
from me_core.perception import MultiModalPerception
from me_core.self_model import SimpleSelfModel
from me_core.tasks.real_tasks import build_real_task_scenarios
from me_core.tasks.runner import run_scenario
from me_core.tools import EchoTool, TimeTool
from me_core.world_model import SimpleWorldModel


class RealTasksScenarioE2ETest(unittest.TestCase):
    def test_run_real_task_scenario(self) -> None:
        scenarios = build_real_task_scenarios()
        if not scenarios:
            self.skipTest("no real tasks available")
        world_model = SimpleWorldModel()
        agent = SimpleAgent(
            perception=MultiModalPerception(),
            world_model=world_model,
            self_model=SimpleSelfModel(),
            drive_system=SimpleDriveSystem(),
            tools={"echo": EchoTool(), "time": TimeTool()},
            learner=SimpleLearner(),
            dialogue_policy=RuleBasedDialoguePolicy(),
            event_stream=EventStream(),
        )
        result = run_scenario(agent, scenarios[0])
        self.assertIsNotNone(result)
        self.assertIn("steps", result.details)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
