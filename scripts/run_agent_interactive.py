"""me-agent 简易交互式命令行演示。

流程：
    - 用户在命令行输入一段文本；
    - 将其作为 MultiModalInput 的 text，封装为感知事件；
    - 更新当前 SelfState 与事件历史；
    - 调用 agent_loop.run_once() 执行一轮主循环（包含感知、学习、自我总结与对话）；
    - 打印本轮的 SelfSummary / InitiativeDecision / message。

运行方式（在仓库根目录）：
    python scripts/run_agent_interactive.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保可以从仓库根目录直接运行本脚本：
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.agent.agent_loop import run_once
from me_core.agent.state_store import StateStore
from me_core.self_model.self_updater import update_from_event
from me_core.perception import encode_to_event
from me_core.types import MultiModalInput


def run_interactive() -> None:
    """启动一个简单的 REPL，与 me-agent 进行多轮交互。"""

    logging.basicConfig(level=logging.INFO)

    print("=== me-agent 交互式演示 ===")  # noqa: T201
    print("输入一段文本，我会感知并自述；输入 `quit` 退出。\n")  # noqa: T201

    # 使用默认状态文件，便于多轮对话共享同一状态
    store = StateStore(path=Path("agent_state.json"))

    try:
        while True:
            try:
                user_text = input("你: ").strip()
            except EOFError:
                break

            if user_text.lower() in {"quit", "exit", "q"}:
                print("再见～")  # noqa: T201
                break

            if not user_text:
                continue

            # 1) 将用户输入封装为多模态感知事件，并更新自我状态与事件历史
            mm_input = MultiModalInput(text=user_text)
            perception_event = encode_to_event(mm_input, source="cli_user_input")

            self_state = store.get_self_state()
            self_state = update_from_event(self_state, perception_event)
            store.set_self_state(self_state)
            store.append_events([perception_event])
            store.add_event_summary("perceive: 收到用户一条文本输入")
            store.save_state()

            # 2) 执行一轮主循环，并打印详细信息
            result = run_once(return_details=True)
            if result is None:
                print("本轮未获得详细信息。")  # noqa: T201
                continue

            self_summary, decision, message = result

            print("\n[SelfSummary]")  # noqa: T201
            for key, value in self_summary.items():
                print(f"  - {key}: {value}")  # noqa: T201

            print("\n[InitiativeDecision]")  # noqa: T201
            print(f"  should_speak: {decision.should_speak}")  # noqa: T201
            print(f"  intent      : {decision.intent}")  # noqa: T201
            print(f"  topic       : {decision.topic}")  # noqa: T201

            print("\n[Message]")  # noqa: T201
            if message:
                print(f"  {message}")  # noqa: T201
            else:
                print("  （本轮未主动发言）")  # noqa: T201

            print("\n---\n")  # noqa: T201
    finally:
        # 结束前简单提示状态文件位置，便于后续调试
        print("当前状态已保存在 agent_state.json 中。")  # noqa: T201


if __name__ == "__main__":
    run_interactive()

