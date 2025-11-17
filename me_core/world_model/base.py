from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from me_core.event_stream import EventHistory
from me_core.types import AgentEvent, EventKind


class BaseWorldModel(ABC):
    """世界模型基类。

    设计意图：
        - 维护对“外部世界”的抽象表示；
        - 以事件流为输入，构建简单的统计或结构化记忆；
        - 为后续引入更复杂的图结构、因果关系建模等留出扩展位。
    """

    @abstractmethod
    def update(self, events: List[AgentEvent]) -> None:
        """使用一批新事件更新世界模型。"""

    @abstractmethod
    def summarize(self) -> Dict[str, Any]:
        """返回当前世界模型的简要摘要，便于对话/决策模块使用。"""


@dataclass
class SimpleWorldModel(BaseWorldModel):
    """基于事件历史的简易世界模型。

    当前只做两类统计：
        - 事件总体分布（委托给 EventHistory）；
        - 工具调用成功率（按工具名聚合）。

    这样可以为驱动力与对话策略提供一个“最近世界状态”的粗略感知。
    """

    history: EventHistory = field(
        default_factory=lambda: EventHistory(max_events=200)
    )
    tool_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def update(self, events: List[AgentEvent]) -> None:
        """将新事件写入历史，并更新工具统计。"""

        if not events:
            return

        self.history.extend(events)

        for e in events:
            kind = (
                e.kind.value
                if isinstance(e.kind, EventKind)
                else (e.kind or e.event_type)
            )
            if kind != EventKind.TOOL_RESULT.value:
                continue

            payload = e.payload or {}
            tool_name = str(payload.get("tool_name") or "unknown_tool")
            success = bool(payload.get("success"))

            stats = self.tool_stats.setdefault(
                tool_name, {"success": 0, "failure": 0}
            )
            if success:
                stats["success"] += 1
            else:
                stats["failure"] += 1

    def summarize(self) -> Dict[str, Any]:
        """生成世界模型的摘要信息。

        返回字典示例：
            {
                "events": {...EventHistory.summarize()},
                "tools": {
                    "echo": {"success": 3, "failure": 0, "success_rate": 1.0},
                    ...
                },
            }
        """

        events_summary = self.history.summarize()
        tools_summary: Dict[str, Dict[str, Any]] = {}

        for name, stats in self.tool_stats.items():
            success = stats.get("success", 0)
            failure = stats.get("failure", 0)
            total = success + failure
            success_rate = success / total if total > 0 else None
            tools_summary[name] = {
                "success": success,
                "failure": failure,
                "success_rate": success_rate,
            }

        return {
            "events": events_summary,
            "tools": tools_summary,
        }

