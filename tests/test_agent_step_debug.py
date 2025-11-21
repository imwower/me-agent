from __future__ import annotations

import unittest

from me_core.agent import SimpleAgent
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.event_stream import EventStream
from me_core.learning import SimpleLearner
from me_core.perception import TextPerception
from me_core.self_model import SimpleSelfModel
from me_core.tools import EchoTool, TimeTool
from me_core.world_model import SimpleWorldModel


class AgentStepDebugTestCase(unittest.TestCase):
    def build_agent(self) -> SimpleAgent:
        perception = TextPerception()
        world_model = SimpleWorldModel()
        self_model = SimpleSelfModel()
        drive_system = SimpleDriveSystem()
        tools = {"echo": EchoTool(), "time": TimeTool()}
        learner = SimpleLearner()
        dialogue_policy = RuleBasedDialoguePolicy()
        event_stream = EventStream()
        return SimpleAgent(
            perception=perception,
            world_model=world_model,
            self_model=self_model,
            drive_system=drive_system,
            tools=tools,
            learner=learner,
            dialogue_policy=dialogue_policy,
            event_stream=event_stream,
            agent_id="test_agent",
        )

    def test_step_with_debug_flag(self) -> None:
        agent = self.build_agent()
        reply = agent.step("你好", debug=True)
        self.assertIsNotNone(reply)
        self.assertIsNotNone(agent.last_intent)
        self.assertGreater(agent.world_model.current_step, 0)


if __name__ == "__main__":
    unittest.main()
