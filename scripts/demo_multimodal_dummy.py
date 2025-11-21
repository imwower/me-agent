"""Dummy 多模态对齐 CLI Demo。

功能：
    - 支持纯文本或“文本 + 图片路径”两种输入模式；
    - 基于 DummyEmbeddingBackend + 概念空间进行占位式多模态对齐；
    - 打印本轮事件、概念对齐结果以及自我模型描述。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# 确保可以从仓库根目录直接运行本脚本
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.agent import SimpleAgent
from me_core.dialogue import RuleBasedDialoguePolicy
from me_core.drives import SimpleDriveSystem
from me_core.event_stream import EventStream
from me_core.learning import SimpleLearner
from me_core.perception import TextPerception
from me_core.self_model import SimpleSelfModel
from me_core.tools import EchoTool, TimeTool
from me_core.world_model import SimpleWorldModel


def build_agent() -> SimpleAgent:
    perception = TextPerception(split_sentences=True)
    world_model = SimpleWorldModel()
    self_model = SimpleSelfModel()
    drive_system = SimpleDriveSystem()
    tools = {
        "echo": EchoTool(),
        "time": TimeTool(),
    }
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
        agent_id="multimodal_dummy_demo",
    )


def _print_step_summary(agent: SimpleAgent, new_events: List) -> None:
    """打印本轮事件与概念对齐结果。"""

    for ev in new_events:
        payload = ev.payload or {}
        brief = ""
        if isinstance(payload, dict):
            if isinstance(payload.get("text"), str):
                brief = payload["text"][:40]
            elif isinstance(payload.get("path"), str):
                brief = payload["path"]
            elif isinstance(payload.get("raw"), dict) and isinstance(payload["raw"].get("text"), str):
                brief = payload["raw"]["text"][:40]
        print(f"[EVENT] {ev.modality}: {brief}")  # noqa: T201

        concept_id = None
        if isinstance(ev.meta, dict):
            concept_id = ev.meta.get("concept_id")

        if concept_id:
            concept = next(
                (c for c in agent.concept_space.all_concepts() if str(c.id) == str(concept_id)),
                None,
            )
            stats = agent.world_model.concept_stats.get(concept.id if concept else concept_id)  # type: ignore[index]
            count = stats.count if stats else 0
            mods = sorted(stats.modalities) if stats else []
            concept_name = concept.name if concept else str(concept_id)
            print(  # noqa: T201
                f"[ALIGN] {ev.modality} -> concept={concept_name} (id={concept_id}) "
                f"count={count} modalities={mods}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy 多模态对齐示例")
    parser.add_argument("--image", type=str, default=None, help="可选：每轮输入附带的图片路径")
    args = parser.parse_args()

    image_path = Path(args.image).as_posix() if args.image else None
    agent = build_agent()

    print("=== me-agent R0 Dummy 多模态对齐 Demo ===")  # noqa: T201
    print("输入任意文本开始；若提供 --image，则每轮都会同时感知该图片路径。")  # noqa: T201
    print("输入 exit/quit 退出。\n")  # noqa: T201

    while True:
        try:
            line = input("你: ").strip()
        except EOFError:
            break

        if not line:
            continue
        if line.lower() in {"exit", "quit", "q"}:
            print("再见～")  # noqa: T201
            break

        before = len(agent.event_stream.to_list())
        reply = agent.step(line, image_path=image_path)
        after_events = agent.event_stream.to_list()
        new_events = after_events[before:]

        _print_step_summary(agent, new_events)
        self_desc = agent.self_model.describe_self(agent.world_model)  # type: ignore[attr-defined]
        print(f"[SELF] {self_desc}")  # noqa: T201
        if reply:
            print(f"[AGENT] {reply}")  # noqa: T201
        else:
            print("[AGENT] （本轮保持沉默）")  # noqa: T201

        print("\n---\n")  # noqa: T201


if __name__ == "__main__":
    main()
