"""展示 me-agent 当前状态与配置的简单脚本。

内容包括：
    - Agent 主循环配置（AgentLoopConfig）
    - 驱动力配置（DrivesConfig）
    - 学习配置（LearningConfig）
    - 当前 SelfState 的关键信息与能力趋势
    - 最近若干条结构化事件摘要

运行方式（在仓库根目录）：
    python scripts/show_agent_status.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保可以从仓库根目录直接运行本脚本：
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.agent.agent_loop import AgentLoopConfig
from me_core.agent.state_store import StateStore
from me_core.drives.config import DEFAULT_DRIVES_CONFIG
from me_core.learning.config import DEFAULT_LEARNING_CONFIG


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    print("=== me-agent 当前状态与配置 ===")  # noqa: T201

    # 1) 配置概览
    loop_cfg = AgentLoopConfig()
    print("\n[Agent 主循环配置]")  # noqa: T201
    print(f"  learning_uncertainty: {loop_cfg.learning_uncertainty}")  # noqa: T201
    print(f"  learning_threshold  : {loop_cfg.learning_threshold}")  # noqa: T201
    print(f"  history_window      : {loop_cfg.history_window}")  # noqa: T201

    drives_cfg = DEFAULT_DRIVES_CONFIG
    print("\n[驱动力配置]")  # noqa: T201
    print(f"  user_command_step         : {drives_cfg.user_command_step}")  # noqa: T201
    print(f"  implicit_smooth_alpha     : {drives_cfg.implicit_smooth_alpha}")  # noqa: T201
    print(f"  high_response_threshold   : {drives_cfg.high_response_threshold}")  # noqa: T201
    print(f"  low_response_threshold    : {drives_cfg.low_response_threshold}")  # noqa: T201
    print(f"  exploration_high_threshold: {drives_cfg.exploration_high_threshold}")  # noqa: T201
    print(f"  learning_success_high_thr : {drives_cfg.learning_success_high_threshold}")  # noqa: T201
    print(f"  learning_adjust_step      : {drives_cfg.learning_adjust_step}")  # noqa: T201

    learn_cfg = DEFAULT_LEARNING_CONFIG
    print("\n[学习配置]")  # noqa: T201
    print(f"  desire_threshold     : {learn_cfg.desire_threshold}")  # noqa: T201
    print(f"  max_knowledge_entries: {learn_cfg.max_knowledge_entries}")  # noqa: T201

    # 2) 自我状态与能力趋势
    state_path = Path("agent_state.json")
    if not state_path.exists():
        print("\n[SelfState] 尚未发现 agent_state.json，可以先运行一次主循环或交互脚本。")  # noqa: T201
        return

    store = StateStore(path=state_path)
    self_state = store.get_self_state()

    print("\n[SelfState 概览]")  # noqa: T201
    print(f"  identity : {self_state.identity}")  # noqa: T201
    print(f"  needs    : {self_state.needs}")  # noqa: T201

    # 展示前若干项能力
    if self_state.capabilities:
        print("\n  能力水平（前若干项）:")  # noqa: T201
        items = sorted(
            self_state.capabilities.items(), key=lambda x: x[1], reverse=True
        )[:5]
        for name, level in items:
            print(f"    - {name}: {level:.3f}")  # noqa: T201
    else:
        print("\n  能力水平：暂无记录。")  # noqa: T201

    # 能力变化趋势
    if self_state.capability_trend:
        print("\n  能力变化趋势:")  # noqa: T201
        for name, delta in self_state.capability_trend.items():
            print(f"    - {name}: Δ={delta:+.3f}")  # noqa: T201
    else:
        print("\n  能力变化趋势：暂无显著变化。")  # noqa: T201

    # 3) 最近事件摘要（近若干条）
    events = store.get_events(limit=5)
    if not events:
        print("\n[最近事件] 尚无结构化事件记录。")  # noqa: T201
        return

    print("\n[最近结构化事件]")  # noqa: T201
    for e in events:
        kind = None
        if isinstance(e.payload, dict):
            kind = e.payload.get("kind")
        print(
            f"  - time={e.timestamp.isoformat()} "
            f"type={e.event_type} kind={kind}"  # noqa: T201
        )


if __name__ == "__main__":
    main()

