from __future__ import annotations

import unittest

from me_core.tasks import Scenario, TaskStep, run_scenario


class _FakeAgent:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def step(self, user_input: str, image_path: str | None = None, debug: bool = False) -> str:
        _ = (user_input, image_path, debug)
        return self.reply


class TasksRunnerTestCase(unittest.TestCase):
    def test_run_scenario_keyword_eval(self) -> None:
        scenario = Scenario(
            id="s1",
            name="test",
            description="",
            steps=[TaskStep(user_input="hi", expected_keywords=["hello"])],
            eval_config={"case_insensitive": True},
        )
        agent = _FakeAgent("Hello world")
        result = run_scenario(agent, scenario)  # type: ignore[arg-type]
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.score, 0.9)


if __name__ == "__main__":
    unittest.main()
