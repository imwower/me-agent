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

from me_core.agent.agent_loop import AgentLoopConfig, run_once
from me_core.agent.state_store import StateStore
from me_core.self_model.self_updater import update_from_event
from me_core.perception import encode_to_event
from me_core.types import MultiModalInput


def _handle_command(line: str, loop_config: AgentLoopConfig) -> bool:
    """处理以 '!' 开头的配置命令。

    返回值：
        True  表示命令已处理，且主循环可以继续；
        False 表示用户希望退出交互。
    """

    parts = line.lstrip("!").strip().split()
    if not parts:
        return True

    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in {"quit", "exit", "q"}:
        print("收到退出命令，再见～")  # noqa: T201
        return False

    if cmd in {"help", "h"}:
        print(
            "可用命令：\n"
            "  !help              查看帮助\n"
            "  !unc <值>          设置学习不确定性 (0~1)\n"
            "  !hist <整数>       设置历史事件窗口大小\n"
            "  !lthr <值>         设置学习意愿阈值 (0~1)\n"
            "  !cfg               查看当前配置\n"
            "  !quit              退出\n"
        )  # noqa: T201
        return True

    if cmd == "cfg":
        print("[当前 Agent 配置]")  # noqa: T201
        print(f"  learning_uncertainty: {loop_config.learning_uncertainty}")  # noqa: T201
        print(f"  learning_threshold  : {loop_config.learning_threshold}")  # noqa: T201
        print(f"  history_window      : {loop_config.history_window}")  # noqa: T201
        return True

    if cmd == "unc" and args:
        try:
            value = float(args[0])
        except ValueError:
            print("  无法解析不确定性数值，请输入 0~1 之间的数字。")  # noqa: T201
            return True
        loop_config.learning_uncertainty = max(0.0, min(1.0, value))
        print(f"  已更新 learning_uncertainty = {loop_config.learning_uncertainty}")  # noqa: T201
        return True

    if cmd == "hist" and args:
        try:
            value = int(args[0])
        except ValueError:
            print("  历史窗口大小需要是整数。")  # noqa: T201
            return True
        loop_config.history_window = max(1, value)
        print(f"  已更新 history_window = {loop_config.history_window}")  # noqa: T201
        return True

    if cmd == "lthr" and args:
        try:
            value = float(args[0])
        except ValueError:
            print("  学习阈值需要是 0~1 之间的数字。")  # noqa: T201
            return True
        loop_config.learning_threshold = max(0.0, min(1.0, value))
        print(f"  已更新 learning_threshold = {loop_config.learning_threshold}")  # noqa: T201
        return True

    print("  未知命令，输入 !help 查看可用选项。")  # noqa: T201
    return True


def run_interactive() -> None:
    """启动一个简单的 REPL，与 me-agent 进行多轮交互。"""

    logging.basicConfig(level=logging.INFO)

    print("=== me-agent 交互式演示 ===")  # noqa: T201
    print("输入一段文本，我会感知并自述；输入 `quit` 退出。\n")  # noqa: T201

    # 使用默认状态文件，便于多轮对话共享同一状态
    store = StateStore(path=Path("agent_state.json"))
    loop_config = AgentLoopConfig()

    try:
        while True:
            try:
                user_text = input("你: ").strip()
            except EOFError:
                break

            # 以 '!' 开头视为配置命令，不进入感知/主循环
            if user_text.startswith("!"):
                if not _handle_command(user_text, loop_config):
                    break
                continue

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
            result = run_once(return_details=True, config=loop_config)
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
