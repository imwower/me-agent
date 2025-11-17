from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from me_core.event_stream import EventStream
from me_core.self_model.self_state import SelfState, default_self_state
from me_core.self_model.self_summarizer import summarize_self
from me_core.self_model.self_updater import update_from_event
from me_core.drives.drive_vector import DriveVector
from me_core.types import AgentEvent, AgentState, EventKind

logger = logging.getLogger(__name__)


@dataclass
class AgentCore:
    """单个智能体的核心状态与最小闭环逻辑。

    本类对应整体架构中的“Single Agent Layer”中最简版本：
        - 持有 SelfState 与 DriveVector，聚合为 AgentState；
        - 使用 EventStream 作为轻量级的事件记忆；
        - 在没有真实环境的情况下，通过 step() 生成虚拟内部事件，
          驱动自我模型更新，并定期输出自我总结。

    在后续 Phase 中，AgentCore 会被扩展为：
        - 接入真实 Env 与 WorldModel；
        - 持有 Memory / ToolSystem / LearningManager 等子模块；
        - 由 AgentLoop 负责在环境中循环调用。
    """

    self_state: SelfState = field(default_factory=default_self_state)
    drives: DriveVector = field(default_factory=DriveVector)
    event_stream: EventStream = field(default_factory=EventStream)
    step_index: int = 0

    def as_agent_state(self) -> AgentState:
        """将内部状态打包为 AgentState 聚合结构。

        当前仅填充 self_state 与 drives，其余字段使用简单占位字典。
        后续在接入 WorldModel / Memory / ToolSystem 时，会将更多信息写入其中。
        """

        return AgentState(
            self_state=self.self_state,
            drives=self.drives,
            world_model_state={},  # Phase1 尚未接入真正世界模型
            memory_state={"event_count": len(self.event_stream.to_list())},
            tool_library_state={},
        )

    def _create_internal_event(self) -> AgentEvent:
        """为 Phase1 Demo 生成一个“内部自述”事件。

        在没有真实环境的前提下，我们通过该事件来驱动自我模型：
            - event_type 使用 EventKind.INTERNAL；
            - payload 中记录当前 step 序号以及简单说明。
        """

        payload = {
            "kind": EventKind.INTERNAL.value,
            "step_index": self.step_index,
            "message": "在虚空环境中进行了一次自我反思步。",
        }
        event = AgentEvent.now(
            event_type=EventKind.INTERNAL.value,
            payload=payload,
            source="agent_internal",
        )
        logger.info("生成内部事件用于自我更新: %s", event.pretty())
        return event

    def step(self) -> AgentEvent:
        """执行一次最小的 Agent 步骤。

        流程：
            1）生成一条内部事件（模拟一次“自我反思”或“内部心跳”）；
            2）将事件写入 EventStream；
            3）调用 update_from_event 更新 SelfState；
            4）步数加一；
            5）返回本次事件，便于上层在需要时进行记录或调试。
        """

        event = self._create_internal_event()
        self.event_stream.append_event(event)
        self.self_state = update_from_event(self.self_state, event)
        self.step_index += 1

        logger.info("完成一步 AgentCore 更新: step_index=%d", self.step_index)
        return event

    def summarize_self(self) -> dict:
        """返回当前自我总结字典。

        该方法直接调用 summarize_self(SelfState)，
        是 Phase1 中“我是谁 / 我能做什么 / 我需要什么”的主要输出入口。
        """

        summary = summarize_self(self.self_state)
        logger.info("当前自我总结: %s", summary)
        return summary

    def recent_events(self, limit: Optional[int] = None) -> List[AgentEvent]:
        """获取最近若干条事件，用于上层调试或后续学习模块接入。"""

        events = self.event_stream.to_list()
        if limit is None or limit >= len(events):
            return events
        return events[-limit:]

