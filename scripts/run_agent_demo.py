"""me-agent 演示脚本。

功能概述：
    - 初始化一个带有默认 SelfState 与 DriveVector 的 StateStore；
    - 连续运行三轮 agent 主循环；
    - 第二轮之前通过 apply_user_command 提升 chat_level / social_need，
      模拟用户说“多陪我聊会儿”，观察输出是否更“健谈”；
    - 在控制台打印每一轮的：
        * 自我总结（self_summary）
        * 对话决策（InitiativeDecision）
        * 最终生成的 message（若有）。

运行方式：
    python scripts/run_agent_demo.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保可以从仓库根目录直接运行本脚本：
# python scripts/run_agent_demo.py
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.agent.agent_loop import run_once
from me_core.agent.state_store import StateStore
from me_core.drives.drive_update import apply_user_command
from me_core.drives.drive_vector import DriveVector
from me_core.self_model.self_state import SelfState


def init_state(store: StateStore) -> None:
    """初始化一个默认的 SelfState 与 DriveVector。"""

    self_state = SelfState(
        identity="一个多模态自我探索的原型智能体",
        capabilities={
            "summarize": 0.7,
            "plan": 0.6,
        },
        needs=["需要更多真实场景的数据和任务"],
    )

    drives = DriveVector(
        chat_level=0.4,
        curiosity_level=0.7,
        exploration_level=0.6,
        learning_intensity=0.7,
        social_need=0.4,
        data_need=0.6,
    )

    store.set_self_state(self_state)
    store.set_drives(drives)
    store.save_state()


def run_demo_round(round_index: int, note: str | None = None) -> None:
    """执行一轮 run_once，并在控制台打印详细信息。"""

    print(f"\n=== 第 {round_index} 轮 ===")  # noqa: T201
    if note:
        print(f"说明：{note}")  # noqa: T201

    result = run_once(return_details=True)
    if result is None:
        print("本轮未获得详细信息。")  # noqa: T201
        return

    self_summary, decision, message = result

    # 打印自我总结
    print("\n[SelfSummary]")  # noqa: T201
    for key, value in self_summary.items():
        print(f"  - {key}: {value}")  # noqa: T201

    # 打印对话决策
    print("\n[InitiativeDecision]")  # noqa: T201
    print(f"  should_speak: {decision.should_speak}")  # noqa: T201
    print(f"  intent      : {decision.intent}")  # noqa: T201
    print(f"  topic       : {decision.topic}")  # noqa: T201

    # 打印最终生成的信息
    print("\n[Message]")  # noqa: T201
    if message:
        print(f"  {message}")  # noqa: T201
    else:
        print("  （本轮未主动发言）")  # noqa: T201


def main() -> None:
    # 配置基础日志，便于在终端观察系统内部行为
    logging.basicConfig(level=logging.INFO)

    # 使用默认路径的 StateStore，这样 agent_loop.run_once() 可以直接读取同一份状态
    state_path = Path("agent_state.json")
    store = StateStore(path=state_path)

    # 初始化默认状态（会覆盖旧状态）
    init_state(store)

    # 第一次：默认驱动力
    run_demo_round(1, note="默认驱动力设置。")

    # 第二次之前：提高聊天与社交相关驱动力，模拟用户说“多陪我聊会儿”
    drives = store.get_drives()
    drives_after = apply_user_command(drives, "多陪我聊天")
    store.set_drives(drives_after)
    store.save_state()

    run_demo_round(2, note="已提升 chat_level / social_need。")

    # 第三次：继续在新的驱动力下运行，观察是否更倾向于主动发言
    run_demo_round(3, note="在提升后的驱动力下再次运行。")


if __name__ == "__main__":
    main()
