from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from me_core.dialogue import DialoguePlanner, InitiativeDecision, generate_message
from me_core.learning.learning_manager import LearningManager
from me_core.self_model.self_summarizer import summarize_self
from me_core.tools.registry import ToolInfo, ToolRegistry

from .state_store import StateStore

logger = logging.getLogger(__name__)


def _build_default_tool_registry() -> ToolRegistry:
    """构造一个带有少量默认工具的注册表。

    这些工具与 LearningManager 的桩实现配合，用于模拟学习行为。
    """

    registry = ToolRegistry()
    registry.register_tool(
        ToolInfo(
            name="search_papers",
            type="knowledge",
            cost=0.3,
            description="检索与当前主题相关的论文与资料（桩实现）。",
            good_for=["paper", "论文", "SNN"],
        )
    )
    registry.register_tool(
        ToolInfo(
            name="run_simulation",
            type="simulation",
            cost=0.5,
            description="针对当前主题运行一次简化模拟实验（桩实现）。",
            good_for=["simulation", "实验", "SNN"],
        )
    )
    return registry


def run_once(
    return_details: bool = False,
) -> Optional[Tuple[Dict[str, str], InitiativeDecision, str]]:
    """执行一轮简单的 agent 主循环。

    步骤：
        1. 加载当前 SelfState 与 DriveVector；
        2. 基于当前时间构造一个简单上下文；
        3. 生成自我总结 self_summary；
        4. 使用 DialoguePlanner 决定是否主动说话；
        5. 若需要说话，则用 generate_message 生成中文输出；
        6. 同时调用 LearningManager.maybe_learn 模拟一次学习过程；
        7. 将状态写回 StateStore。
    """

    store = StateStore()
    self_state = store.get_self_state()
    drives = store.get_drives()

    now = datetime.now(timezone.utc)
    context = {
        "time_iso": now.isoformat(),
        "topic": "自我介绍与当前学习状态",
        "source": "agent_loop",
    }

    logger.info("开始执行一轮 agent 主循环，当前时间: %s", context["time_iso"])

    # 生成自我总结
    self_summary = summarize_self(self_state)

    # 对话决策与生成
    planner = DialoguePlanner()
    decision = planner.decide_initiative(drives, self_summary, context)

    message = ""
    if decision.should_speak:
        message = generate_message(decision, self_summary, context)
        if message:
            # 日志与标准输出都记录，方便在不同环境中观察
            logger.info("Agent 主动发言: %s", message)
            print(message)  # noqa: T201
            store.add_event_summary(f"say: {message}")
    else:
        logger.info("本轮决策为保持沉默。")
        store.add_event_summary("silent")

    # 模拟一次学习过程：不确定性简单设为 0.6
    registry = _build_default_tool_registry()
    learning_manager = LearningManager(registry=registry)

    learning_results = learning_manager.maybe_learn(
        uncertainty=0.6,
        drives=drives,
        context=context,
    )

    if learning_results:
        store.add_event_summary(
            f"learn: 调用工具 {len(learning_results)} 次，成功 {sum(1 for r in learning_results if r.success)} 次"
        )
    else:
        store.add_event_summary("learn: skipped")

    # 将状态写回存储（当前逻辑中 self_state / drives 未改变，但为未来扩展留接口）
    store.set_self_state(self_state)
    store.set_drives(drives)
    store.save_state()

    logger.info("本轮 agent 主循环结束。")

    if return_details:
        return self_summary, decision, message
