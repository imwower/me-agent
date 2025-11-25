import unittest

from me_core.config import AgentConfig
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives.base import Intent
from me_core.learning import SimpleLearner
from me_core.self_model import SimpleSelfModel
from me_core.world_model import SimpleWorldModel


class _StubLLM:
    def __init__(self, fail: bool = False, text: str = "LLM 回复") -> None:
        self.fail = fail
        self.text = text

    def generate_reply(self, prompt: str, meta=None) -> str:
        if self.fail:
            raise RuntimeError("stub failure")
        return self.text


class DialogueLLMFallbackTest(unittest.TestCase):
    def test_llm_path_used_when_enabled(self) -> None:
        cfg = AgentConfig(use_llm_dialogue=True)
        policy = RuleBasedDialoguePolicy(agent_config=cfg, dialogue_llm=_StubLLM())
        reply = policy.generate_reply(
            events=[],
            intent=Intent(kind="reply", priority=1, explanation="test llm"),
            world=SimpleWorldModel(),
            self_model=SimpleSelfModel(),
            learner=SimpleLearner(),
        )
        self.assertEqual(reply, "LLM 回复")

    def test_fallback_to_rule_based_on_failure(self) -> None:
        cfg = AgentConfig(use_llm_dialogue=True)
        policy = RuleBasedDialoguePolicy(agent_config=cfg, dialogue_llm=_StubLLM(fail=True))
        reply = policy.generate_reply(
            events=[],
            intent=Intent(kind="reply", priority=1, explanation="test fallback"),
            world=SimpleWorldModel(),
            self_model=SimpleSelfModel(),
            learner=SimpleLearner(),
        )
        self.assertIsNotNone(reply)
        assert reply is not None
        self.assertIn("【我做】", reply)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
