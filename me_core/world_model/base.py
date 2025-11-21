from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from me_core.alignment.concepts import ConceptId, ConceptNode, ConceptSpace
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
class ConceptStats:
    count: int = 0
    modalities: Set[str] = field(default_factory=set)


@dataclass
class SimpleWorldModel(BaseWorldModel):
    """基于事件历史的简易世界模型。

    当前只做两类统计：
        - 事件总体分布（委托给 EventHistory）；
        - 工具调用成功率（按工具名聚合）。

    这样可以为驱动力与对话策略提供一个“最近世界状态”的粗略感知。
    """

    history: EventHistory = field(default_factory=lambda: EventHistory(max_events=200))
    tool_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    concept_space: ConceptSpace = field(default_factory=ConceptSpace)
    concept_stats: Dict[ConceptId, ConceptStats] = field(default_factory=dict)

    def update(self, events: List[AgentEvent]) -> None:
        """将新事件写入历史，并更新工具统计。"""

        if not events:
            return

        for event in events:
            self.observe_event(event, None)

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

        concept_summary: Dict[str, Dict[str, Any]] = {}
        for cid, stats in self.concept_stats.items():
            concept_summary[str(cid)] = {
                "count": stats.count,
                "modalities": sorted(stats.modalities),
            }

        return {
            "events": events_summary,
            "tools": tools_summary,
            "concepts": concept_summary,
        }

    # 新增：概念相关观测接口 --------------------------------------------------------

    def observe_event(self, event: AgentEvent, concept: Optional[ConceptNode]) -> None:
        """
        在原有事件统计基础上，额外更新 concept_stats。
        """

        self.history.add(event)

        kind = (
            event.kind.value
            if isinstance(event.kind, EventKind)
            else (event.kind or event.event_type)
        )
        if kind == EventKind.TOOL_RESULT.value:
            payload = event.payload or {}
            tool_name = str(payload.get("tool_name") or "unknown_tool")
            success = bool(payload.get("success"))
            stats = self.tool_stats.setdefault(tool_name, {"success": 0, "failure": 0})
            if success:
                stats["success"] += 1
            else:
                stats["failure"] += 1

        concept_node = concept
        if concept_node is None and isinstance(event.meta, dict):
            cid = event.meta.get("concept_id")
            if cid is not None:
                concept_node = next(
                    (c for c in self.concept_space.all_concepts() if str(c.id) == str(cid)),
                    None,
                )

        if concept_node is not None:
            stats = self.concept_stats.setdefault(concept_node.id, ConceptStats())
            stats.count += 1
            stats.modalities.add(event.modality or "unknown")

    def get_concept_stats(self, concept: ConceptNode) -> ConceptStats | None:
        return self.concept_stats.get(concept.id)

    def recent_concepts(self, top_k: int = 10) -> List[ConceptNode]:
        ordered = sorted(
            self.concept_stats.items(),
            key=lambda item: item[1].count,
            reverse=True,
        )
        ids = [cid for cid, _ in ordered[:top_k]]
        concepts: List[ConceptNode] = []
        for cid in ids:
            for c in self.concept_space.all_concepts():
                if str(c.id) == str(cid):
                    concepts.append(c)
                    break
        return concepts
