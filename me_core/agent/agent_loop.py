from __future__ import annotations

import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from me_core.dialogue import DialoguePlanner, InitiativeDecision, generate_message
from me_core.drives.drive_update import implicit_adjust
from me_core.learning.config import DEFAULT_LEARNING_CONFIG
from me_core.learning.learning_manager import LearningManager
from me_core.perception import encode_to_event
from me_core.self_model.self_summarizer import summarize_self
from me_core.self_model.self_updater import aggregate_stats, update_from_event
from me_core.tools.registry import ToolInfo, ToolRegistry
from me_core.types import AgentEvent, MultiModalInput

from .state_store import StateStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentLoopConfig:
    """Agent 主循环的可调配置项。

    目前仅包含少量关键超参数，后续如有需要可以逐步扩展：
        - learning_uncertainty: 学习阶段使用的“不确定性”估计值；
        - history_window: 聚合历史能力统计时使用的事件窗口大小。
    """

    learning_uncertainty: float = 0.6
    history_window: int = 100
    learning_threshold: float = DEFAULT_LEARNING_CONFIG.desire_threshold


DEFAULT_AGENT_LOOP_CONFIG = AgentLoopConfig()


def _select_focus_topic(self_state) -> str:
    """根据自我状态自动选择本轮学习关注的主题。

    策略（从高到低优先级）：
        1. 若存在明确的局限（limitations），优先围绕第一条局限进行学习；
        2. 若存在能力明显下降（capability_trend 为负且幅度较大），
           则围绕下降最明显的能力进行“补救性”学习；
        3. 否则返回一个泛化主题，表示通用的自我改进。
    """

    # 1) 优先针对自我局限进行学习
    if self_state.limitations:
        return f"改善自身局限：{self_state.limitations[0]}"

    # 2) 其次针对最近明显下降的能力进行自我改进
    negative_trend = {
        name: delta
        for name, delta in self_state.capability_trend.items()
        if delta < 0.0
    }
    if negative_trend:
        # 选取下降幅度最大的能力（delta 最小）
        target_capability = min(negative_trend, key=negative_trend.get)
        return f"提高能力「{target_capability}」的表现"

    # 3) 默认的通用学习主题
    return "通用学习与自我改进"


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
    config: Optional[AgentLoopConfig] = None,
) -> Optional[Tuple[Dict[str, str], InitiativeDecision, str]]:
    """执行一轮简单的 agent 主循环。

    步骤：
        1. 加载当前 SelfState 与 DriveVector；
        2. 基于当前时间构造一个简单上下文；
        3. 生成自我总结 self_summary；
        4. 使用 DialoguePlanner 决定是否主动说话；
        5. 若需要说话，则用 generate_message 生成中文输出；
        6. 同时调用 LearningManager.maybe_learn 模拟一次学习过程；
        7. 根据学习结果构造 AgentEvent，更新 SelfState 与 Drives；
        8. 将状态写回 StateStore。
    """

    if config is None:
        config = DEFAULT_AGENT_LOOP_CONFIG

    store = StateStore()
    self_state = store.get_self_state()
    drives = store.get_drives()

    now = datetime.now(timezone.utc)

    # 基于当前自我状态自动选择一个“本轮关注主题”，作为学习与感知的锚点
    focus_topic = _select_focus_topic(self_state)
    context = {
        "time_iso": now.isoformat(),
        "topic": focus_topic,
        "source": "agent_loop",
    }

    logger.info("开始执行一轮 agent 主循环，当前时间: %s, 主题: %s", context["time_iso"], focus_topic)

    # 先模拟一次简单的多模态感知，将其作为事件写入自我模型与事件流
    perception_input = MultiModalInput(
        text=f"当前时间 {context['time_iso']}，当前主题：{context['topic']}",
    )
    perception_event = encode_to_event(perception_input, source="agent_loop")
    self_state = update_from_event(self_state, perception_event)
    store.append_events([perception_event])
    store.add_event_summary("perceive: 多模态感知了一次当前环境信息")

    # 模拟一次学习过程：不确定性简单设为 0.6
    registry = _build_default_tool_registry()
    # 使用状态存储中已有的知识库初始化学习管理器，实现跨轮持久化
    learning_manager = LearningManager(
        registry=registry, knowledge_base=store.get_knowledge_base()
    )

    learning_results = learning_manager.maybe_learn(
        uncertainty=config.learning_uncertainty,
        drives=drives,
        context=context,
        threshold=config.learning_threshold,
    )

    if learning_results:
        store.add_event_summary(
            f"learn: 调用工具 {len(learning_results)} 次，成功 {sum(1 for r in learning_results if r.success)} 次"
        )

        # 将学习结果映射为标准的 AgentEvent，并驱动自我模型更新
        tool_events: List[AgentEvent] = []
        success_count = 0
        for result in learning_results:
            if result.success:
                success_count += 1

            payload: Dict[str, object] = {
                "kind": "task",
                "task_type": result.tool_name,
                "success": result.success,
                "topic": context.get("topic"),
            }
            # 预留错误信息字段，方便未来真实工具失败时记录局限
            details = getattr(result, "details", None)
            if (not result.success) and isinstance(details, dict) and details.get("message"):
                payload["error"] = str(details["message"])

            event = AgentEvent.now(event_type="task", payload=payload)
            tool_events.append(event)
            self_state = update_from_event(self_state, event)

        logger.info(
            "根据学习结果生成任务事件并更新自我状态: events=%s",
            tool_events,
        )

        # 将新事件加入状态存储，并基于历史事件重新聚合能力与局限
        store.append_events(tool_events)
        history = store.get_events(limit=config.history_window)
        self_state = aggregate_stats(self_state, history)
        logger.info("基于最近 %d 条事件聚合后的自我状态: %s", len(history), self_state)

        # 使用学习成功率作为驱动力隐式调整的输入
        success_ratio = success_count / len(learning_results)
        drives = implicit_adjust(
            drives,
            {"learning_success": success_ratio},
        )
        logger.info(
            "根据学习成功率隐式调整驱动力: success_ratio=%.3f, drives=%s",
            success_ratio,
            drives,
        )

        # 将最近学习内容加入上下文，便于对话模块在需要时引用
        recent_knowledge = learning_manager.query_knowledge(
            topic=context.get("topic", ""),
            max_results=3,
        )
        context["has_recent_learning"] = bool(recent_knowledge)
        context["recent_knowledge"] = recent_knowledge
    else:
        store.add_event_summary("learn: skipped")
        context["has_recent_learning"] = False
        context["recent_knowledge"] = []

    # 生成自我总结（此时已纳入感知与学习带来的自我更新）
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

    # 将状态写回存储
    store.set_self_state(self_state)
    store.set_drives(drives)
    store.set_knowledge_base(learning_manager.knowledge_base)
    store.save_state()

    logger.info("本轮 agent 主循环结束。")

    if return_details:
        return self_summary, decision, message
