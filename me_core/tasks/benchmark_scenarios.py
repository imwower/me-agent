from __future__ import annotations

from typing import List

from .types import Scenario, TaskStep


def benchmark_multimodal_small() -> Scenario:
    """一个小型多模态基准场景，占位使用 tests/data/dummy.png。"""

    return Scenario(
        id="vqa_small",
        name="VQA Small",
        description="小型中文 VQA 基准（占位）",
        steps=[
            TaskStep(
                user_input="这张图片是什么？请用一句话描述。",
                image_path="tests/data/dummy.png",
                expected_keywords=["图片", "概念"],
                eval_config={"mode": "contains_any"},
            )
        ],
        eval_config={"case_insensitive": True},
        requires_brain_infer=False,
    )


def list_benchmark_scenarios() -> List[Scenario]:
    return [benchmark_multimodal_small()]
