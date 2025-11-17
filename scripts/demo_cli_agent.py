"""me-agent 最小可运行 Agent 循环演示脚本。

功能：
    - 构造一个由 SimpleAgent 及默认组件组成的代理人；
    - 在命令行中循环读取用户输入文本；
    - 对每一轮输入执行“感知 → 意图 → 工具调用（可选）→ 对话回复”的闭环；
    - 在终端以中文打印内部思考日志（我想 / 我要 / 我做）和最终回复。

运行方式（在仓库根目录）：
    python scripts/demo_cli_agent.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def build_default_agent() -> SimpleAgent:
    """构造一个带有默认组件的 SimpleAgent 实例。

    默认组件包括：
        - TextPerception：将用户文本输入转为感知事件；
        - SimpleWorldModel：基于事件历史的世界模型；
        - SimpleSelfModel：基于 SelfState 的自我模型；
        - SimpleDriveSystem：根据最近事件给出“回复 / 保持安静”等意图；
        - EchoTool / TimeTool：两个简单工具，用于展示工具调用流程；
        - SimpleLearner：观察事件计数的极简学习器；
        - RuleBasedDialoguePolicy：将 Intent 转为“我想 / 我要 / 我做”式回复。
    """

    perception = TextPerception()
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

    agent = SimpleAgent(
        perception=perception,
        world_model=world_model,
        self_model=self_model,
        drive_system=drive_system,
        tools=tools,
        learner=learner,
        dialogue_policy=dialogue_policy,
        event_stream=event_stream,
        agent_id="demo_cli_agent",
    )
    return agent


def main() -> None:
    """启动命令行 demo，与 SimpleAgent 进行多轮对话。"""

    agent = build_default_agent()

    print("=== me-agent 简易对话 demo ===")  # noqa: T201
    print(
        "输入任意文本开始对话，输入 'time' 可以触发时间工具，输入 exit/quit 退出。\n"  # noqa: T201
    )

    while True:
        try:
            user_text = input("你: ").strip()
        except EOFError:
            break

        if not user_text:
            continue

        if user_text.lower() in {"exit", "quit", "q"}:
            print("再见～")  # noqa: T201
            break

        # 一个简单示例：用户输入 "time" 时，驱动力系统仍然返回“回复”意图，
        # 但在后续可以扩展为：根据文本内容构造 call_tool 型 Intent。
        reply = agent.step(user_text)

        if reply:
            print(f"Agent: {reply}")  # noqa: T201
        else:
            print("Agent: （本轮保持沉默）")  # noqa: T201

        print("\n---\n")  # noqa: T201


if __name__ == "__main__":
    main()

