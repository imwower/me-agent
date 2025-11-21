from __future__ import annotations

from typing import Dict, List

from .types import Scenario, TaskStep


class ScenarioRegistry:
    def __init__(self) -> None:
        self._scenarios: Dict[str, Scenario] = {}
        for s in default_scenarios():
            self.register(s)

    def register(self, scenario: Scenario) -> None:
        self._scenarios[scenario.id] = scenario

    def get(self, scenario_id: str) -> Scenario | None:
        return self._scenarios.get(scenario_id)

    def list_ids(self) -> List[str]:
        return list(self._scenarios.keys())


def default_scenarios() -> List[Scenario]:
    return [
        Scenario(
            id="ask_time",
            name="时间询问场景",
            description="多轮询问当前时间，考察工具调用。",
            steps=[
                TaskStep(user_input="现在几点"),
                TaskStep(user_input="请再说一次当前时间"),
            ],
            eval_config={"case_insensitive": True},
        ),
        Scenario(
            id="self_intro",
            name="自我介绍",
            description="询问身份与能力，考察自述质量。",
            steps=[
                TaskStep(user_input="你是谁？", expected_keywords=["我", "能力", "模态"]),
                TaskStep(user_input="你最近在忙什么？", expected_keywords=["忙", "做"]),
            ],
            eval_config={"case_insensitive": True},
        ),
        Scenario(
            id="multimodal_hint",
            name="多模态提示",
            description="给出图片路径并询问，期待模型提示需要真实图片理解。",
            steps=[
                TaskStep(
                    user_input="看看这张图片说说看",
                    image_path="tests/data/dummy.png",
                    expected_keywords=["图片", "概念"],
                )
            ],
            eval_config={"case_insensitive": True},
        ),
    ]


__all__ = ["ScenarioRegistry", "default_scenarios"]
