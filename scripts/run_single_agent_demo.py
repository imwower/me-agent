"""单智能体自述 Demo（无环境版）。

功能概述：
    - 构建一个仅包含 SelfState + DriveVector + 事件流的 AgentCore；
    - 在“虚空环境”中连续运行若干步 step()；
    - 每隔几步调用 summarize_self，打印「我是谁 / 我能做什么 / 我需要什么」；
    - 通过 logging 观察内部事件与自我状态的变化。

运行方式（在仓库根目录）：
    python scripts/run_single_agent_demo.py           # 默认运行 10 步
    python scripts/run_single_agent_demo.py 20        # 运行 20 步
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保可以从仓库根目录直接运行本脚本：
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.agent import AgentCore


def run_demo(num_steps: int = 10, summary_interval: int = 3) -> None:
    """在无环境场景下，运行若干步 AgentCore 自我更新并打印自述。"""

    logging.basicConfig(level=logging.INFO)

    core = AgentCore()

    print(f"=== 单智能体自述 Demo（共 {num_steps} 步）===")  # noqa: T201
    print("说明：每一步都会生成一条内部事件，驱动自我模型更新。")  # noqa: T201
    print("      每隔若干步会打印一次 summarize_self 的结果。\n")  # noqa: T201

    for step in range(1, num_steps + 1):
        core.step()

        if step % summary_interval == 0 or step == num_steps:
            summary = core.summarize_self()
            print(f"\n--- 第 {step} 步自我总结 ---")  # noqa: T201
            print(f"who_am_i        : {summary.get('who_am_i', '')}")  # noqa: T201
            print(f"what_can_i_do   : {summary.get('what_can_i_do', '')}")  # noqa: T201
            print(f"what_do_i_need  : {summary.get('what_do_i_need', '')}")  # noqa: T201


def main() -> None:
    if len(sys.argv) >= 2:
        try:
            num_steps = int(sys.argv[1])
        except ValueError:
            print("参数需要是整数步数，例如: python scripts/run_single_agent_demo.py 20")  # noqa: T201
            return
    else:
        num_steps = 10

    if num_steps <= 0:
        print("步数需要是正整数。")  # noqa: T201
        return

    run_demo(num_steps=num_steps)


if __name__ == "__main__":
    main()

