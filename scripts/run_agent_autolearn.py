"""me-agent 自主学习循环脚本。

功能：
    - 不依赖任何用户输入，连续运行多轮 agent 主循环；
    - 每一轮根据当前 SelfState 自动选择学习主题；
    - 持续积累事件与知识库，并在控制台展示能力与知识变化。

用法（在仓库根目录）：
    python scripts/run_agent_autolearn.py            # 默认运行 5 轮
    python scripts/run_agent_autolearn.py 10         # 运行 10 轮
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


def run_autolearn(num_rounds: int) -> None:
    """运行若干轮自主学习循环，并在每轮结束后打印状态摘要。"""

    logging.basicConfig(level=logging.INFO)

    state_path = Path("agent_state.json")
    # 使用一组更“积极”的学习配置：
    # - 更高的不确定性估计（0.9），鼓励更多主动学习；
    # - 更低的学习意愿阈值（0.05），降低触发门槛；
    # 这样在完全自主场景下更容易看到能力与知识的积累。
    loop_config = AgentLoopConfig(
        learning_uncertainty=0.9,
        learning_threshold=0.05,
    )

    print(f"=== me-agent 自主学习循环（共 {num_rounds} 轮）===")  # noqa: T201
    print("状态将持久化到 agent_state.json，可用 scripts/show_agent_status.py 查看。\n")  # noqa: T201

    # 确保状态文件存在
    store = StateStore(path=state_path)
    store.save_state()

    for i in range(1, num_rounds + 1):
        print(f"\n=== 第 {i} 轮 ===")  # noqa: T201

        # 执行一轮主循环（包含感知、学习、自我总结与对话）
        run_once(return_details=False, config=loop_config)

        # 重新加载最新状态
        store = StateStore(path=state_path)
        self_state = store.get_self_state()
        kb = store.get_knowledge_base()

        # 打印能力与趋势摘要
        print("\n[能力概览]")  # noqa: T201
        if self_state.capabilities:
            items = sorted(
                self_state.capabilities.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for name, level in items:
                print(f"  - {name}: {level:.3f}")  # noqa: T201
        else:
            print("  （当前尚无能力记录）")  # noqa: T201

        print("\n[能力变化趋势]")  # noqa: T201
        if self_state.capability_trend:
            for name, delta in self_state.capability_trend.items():
                print(f"  - {name}: Δ={delta:+.3f}")  # noqa: T201
        else:
            print("  （最近几轮能力变化不明显）")  # noqa: T201

        # 打印知识库大小与最近条目摘要
        print("\n[知识库]")  # noqa: T201
        print(f"  当前条目数: {len(kb)}")  # noqa: T201
        if kb:
            print("  最近条目（最多 3 条）：")  # noqa: T201
            for entry in kb[-3:]:
                tool_name = entry.get("tool_name") or "未知工具"
                topic = entry.get("topic") or "未知主题"
                summary = entry.get("summary") or ""
                print(
                    f"    - tool={tool_name} topic={topic} summary={summary}"  # noqa: T201
                )

        print("\n---")  # noqa: T201


def main() -> None:
    if len(sys.argv) >= 2:
        try:
            num_rounds = int(sys.argv[1])
        except ValueError:
            print("参数需要是整数轮数，例如: python scripts/run_agent_autolearn.py 10")  # noqa: T201
            return
    else:
        num_rounds = 5

    if num_rounds <= 0:
        print("轮数需要是正整数。")  # noqa: T201
        return

    run_autolearn(num_rounds)


if __name__ == "__main__":
    main()
