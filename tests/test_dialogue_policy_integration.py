from __future__ import annotations

import unittest

from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import Intent
from me_core.learning import SimpleLearner
from me_core.self_model import SimpleSelfModel
from me_core.types import AgentEvent, EventKind
from me_core.world_model import SimpleWorldModel


class DialoguePolicyIntegrationTestCase(unittest.TestCase):
    def test_reply_uses_self_description(self) -> None:
        policy = RuleBasedDialoguePolicy()
        world = SimpleWorldModel()
        self_model = SimpleSelfModel()
        learner = SimpleLearner()

        world.advance_step()
        event = AgentEvent.now(
            event_type=EventKind.PERCEPTION.value,
            payload={"raw": {"text": "你好"}},
            kind=EventKind.PERCEPTION,
            source="human",
        )
        world.append_event(event)
        self_model.observe_event(event, step=1)

        intent = Intent(kind="reply", priority=5, explanation="测试生成回复")
        reply = policy.generate_reply(
            events=[event],
            intent=intent,
            world=world,
            self_model=self_model,
            learner=learner,
        )
        self.assertIn("你刚才说", reply)
        self.assertIn("【我想】测试生成回复", reply)
        self.assertIn("【我做】", reply)

    def test_curiosity_reply(self) -> None:
        policy = RuleBasedDialoguePolicy()
        world = SimpleWorldModel()
        self_model = SimpleSelfModel()
        learner = SimpleLearner()

        intent = Intent(
            kind="curiosity",
            priority=3,
            preferred_modality="image",
            extra={"concept_name": "苹果"},
        )
        reply = policy.generate_reply([], intent, world, self_model, learner)
        self.assertIn("好奇", reply)
        self.assertIn("苹果", reply)


if __name__ == "__main__":
    unittest.main()
