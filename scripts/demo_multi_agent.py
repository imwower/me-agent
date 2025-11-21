"""简单的多 Agent 对话演示脚本。

运行方式（仓库根目录）：
    python scripts/demo_multi_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.agent import SimpleAgent  # noqa: E402
from me_core.agent_multi import AgentShell, ConversationHub  # noqa: E402
from me_core.dialogue import RuleBasedDialoguePolicy  # noqa: E402
from me_core.drives import SimpleDriveSystem  # noqa: E402
from me_core.event_stream import EventStream  # noqa: E402
from me_core.learning import SimpleLearner  # noqa: E402
from me_core.perception import TextPerception  # noqa: E402
from me_core.self_model import SimpleSelfModel  # noqa: E402
from me_core.tools import (  # noqa: E402
    EchoTool,
    FileReadTool,
    HttpGetTool,
    SelfDescribeTool,
    TimeTool,
)
from me_core.world_model import SimpleWorldModel  # noqa: E402


def build_agent(agent_id: str) -> SimpleAgent:
    perception = TextPerception()
    world_model = SimpleWorldModel()
    self_model = SimpleSelfModel()
    drive_system = SimpleDriveSystem()
    tools = {
        "echo": EchoTool(),
        "time": TimeTool(),
        "http_get": HttpGetTool(),
        "file_read": FileReadTool(),
        "self_describe": SelfDescribeTool(self_model=self_model, world_model=world_model),
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
        agent_id=agent_id,
    )


def main() -> None:
    agent_a = build_agent("agent_a")
    agent_b = build_agent("agent_b")
    hub = ConversationHub(
        [
            AgentShell(id="agent_a", agent=agent_a),
            AgentShell(id="agent_b", agent=agent_b),
        ]
    )

    script = [
        ("agent_a", "你好，我想了解一下你最近的观察。"),
        ("agent_b", "你有什么工具能力？"),
        ("agent_a", "现在几点了？"),
    ]

    for speaker, message in script:
        print(f"\n[{speaker}] -> {message}")  # noqa: T201
        responses = hub.run_turn(speaker, message)
        for agent_id, reply in responses.items():
            print(f"{agent_id}: {reply}")  # noqa: T201


if __name__ == "__main__":
    main()
