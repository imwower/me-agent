from __future__ import annotations

import unittest

from me_core.agent import SimpleAgent
from me_core.agent_multi import AgentShell, ConversationHub
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.event_stream import EventStream
from me_core.learning import SimpleLearner
from me_core.perception import TextPerception
from me_core.self_model import SimpleSelfModel
from me_core.tools import EchoTool, TimeTool
from me_core.world_model import SimpleWorldModel


def build_agent(agent_id: str) -> SimpleAgent:
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
        agent_id=agent_id,
    )


class MultiAgentConversationTestCase(unittest.TestCase):
    def test_conversation_hub(self) -> None:
        hub = ConversationHub(
            [
                AgentShell("a", build_agent("a")),
                AgentShell("b", build_agent("b")),
            ]
        )
        responses = hub.run_turn("a", "你好")
        self.assertIn("a", responses)
        self.assertIn("b", responses)


if __name__ == "__main__":
    unittest.main()
