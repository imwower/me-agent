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
            id="image_alignment_basic",
            name="图片对齐基础",
            description="给定图片和问题，测试概念对齐与回答。",
            steps=[
                TaskStep(
                    user_input="这张图片里大概是什么？",
                    image_path="examples/apple.png",
                    expected_keywords=["苹果", "水果"],
                    eval_config={"mode": "contains_any"},
                )
            ],
            eval_config={"case_insensitive": True},
        ),
        Scenario(
            id="image_vs_text_consistency",
            name="图文一致性检查",
            description="文本暗示猫，图片是猫，问是不是狗，期待否定或纠正。",
            steps=[
                TaskStep(
                    user_input="我给你一张有猫的图片，这张图片里的动物是狗吗？",
                    image_path="examples/cat.png",
                    expected_keywords=["不", "猫", "不是狗"],
                    eval_config={"mode": "contains_any"},
                )
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
        Scenario(
            id="brain_guided_decision",
            name="SNN 引导策略决策场景",
            description="在回答前调用 self-snn 在线脑推理，根据脑状态调整回答风格。",
            steps=[
                TaskStep(
                    user_input="你现在处于信息不足、不确定性较高的环境，请先思考再给出策略建议。",
                    expected_keywords=["探索", "收集信息", "先观察"],
                    eval_config={"mode": "contains_any"},
                )
            ],
            eval_config={"case_insensitive": True},
            requires_brain_infer=True,
        ),
    ]


__all__ = ["ScenarioRegistry", "default_scenarios"]
