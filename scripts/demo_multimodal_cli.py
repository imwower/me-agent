#!/usr/bin/env python
from __future__ import annotations

"""多模态 CLI Demo。

功能：
    - 构造一个带有多模态感知与概念对齐能力的 SimpleAgent；
    - 支持纯文本输入以及“描述图片”的指令；
    - 每轮打印感知事件、对齐到的概念与中文回复。

用法示例（在仓库根目录）：

    python scripts/demo_multimodal_cli.py

交互示例：

    你: 你好
    你: describe ./some_image.png 这张图片里大概是什么？

当前版本使用 DummyEmbeddingBackend 与占位式多模态工具，不依赖真实视觉模型。
"""

import sys
from pathlib import Path
from typing import Any, Dict

# 确保可以从仓库根目录直接运行本脚本
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.agent import SimpleAgent
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.event_stream import EventStream
from me_core.learning import SimpleLearner
from me_core.perception import MultiModalPerception
from me_core.self_model import SimpleSelfModel
from me_core.tools import EchoTool, TimeTool, MultimodalQATool
from me_core.world_model import SimpleWorldModel
from me_core.types import MultiModalInput


def build_multimodal_agent() -> SimpleAgent:
    perception = MultiModalPerception()
    world_model = SimpleWorldModel()
    self_model = SimpleSelfModel()
    drive_system = SimpleDriveSystem()
    tools: Dict[str, Any] = {
        "echo": EchoTool(),
        "time": TimeTool(),
        "multimodal_qa": MultimodalQATool(),
    }
    learner = SimpleLearner()
    dialogue_policy = RuleBasedDialoguePolicy()
    event_stream = EventStream()

    agent = SimpleAgent(
        perception=perception,
        world_model=world_model,
        self_model=self_model,
        drive_system=drive_system,
        tools=tools,
        learner=learner,
        dialogue_policy=dialogue_policy,
        event_stream=event_stream,
        agent_id="multimodal_cli_agent",
    )
    return agent


def parse_user_input(line: str) -> Any:
    """解析用户输入。

    规则：
        - 以 'describe ' 开头：视为多模态指令，格式：
            describe <image_path> <question...>
        - 否则：视为普通文本输入。
    """

    stripped = line.strip()
    if not stripped:
        return ""

    if stripped.startswith("describe "):
        parts = stripped.split(maxsplit=2)
        if len(parts) < 3:
            return stripped  # 退回普通文本
        _, img_path_str, question = parts
        img_path = Path(img_path_str)
        # 这里不强制要求文件存在，交由后续流程处理
        mm = MultiModalInput(
            text=question,
            image_meta={
                "path": str(img_path),
            },
        )
        return mm

    return stripped


def main() -> None:
    agent = build_multimodal_agent()

    print("=== me-agent 多模态对齐 CLI Demo ===")  # noqa: T201
    print("说明：")  # noqa: T201
    print("  - 直接输入中文文本进行对话；")  # noqa: T201
    print("  - 使用指令 `describe <图片路径> <问题>` 触发简单的文本+图像多模态感知；")  # noqa: T201
    print("  - 输入 exit/quit 退出。\n")  # noqa: T201

    while True:
        try:
            line = input("你: ")
        except EOFError:
            break

        if not line:
            continue
        if line.lower().strip() in {"exit", "quit", "q"}:
            print("再见～")  # noqa: T201
            break

        parsed_input = parse_user_input(line)
        reply = agent.step(parsed_input)

        # 打印最近一次感知事件与概念对齐信息
        events = agent.event_stream.to_list()
        if events:
            last_event = events[-1]
            payload = last_event.payload or {}
            concept_id = last_event.meta.get("concept_id")
            print(f"[调试] 最近事件: {last_event}")  # noqa: T201
            if concept_id and hasattr(agent.world_model, "concept_stats"):
                stats = getattr(agent.world_model, "concept_stats", {})
                concept_info = stats.get(str(concept_id))
                if concept_info:
                    print(  # noqa: T201
                        f"[调试] 对齐到概念: id={concept_id}, "
                        f"name={concept_info.get('name')}, "
                        f"count={concept_info.get('count')}"
                    )
            if "embeddings" in payload:
                mods = ", ".join(payload.get("modalities") or [])
                print(f"[调试] 感知模态: {mods}")  # noqa: T201

        if reply:
            print(f"Agent: {reply}")  # noqa: T201
        else:
            print("Agent: （本轮保持沉默）")  # noqa: T201

        print("\n---\n")  # noqa: T201


if __name__ == "__main__":
    main()
