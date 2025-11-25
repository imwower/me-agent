from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from me_core.alignment.concepts import ConceptId, ConceptNode, ConceptSpace
from me_core.event_stream import EventHistory
from me_core.types import AgentEvent, EventKind
from me_core.brain import BrainSnapshot


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
    modalities: Dict[str, int] = field(default_factory=dict)
    last_seen_step: int = -1


@dataclass
class TimedEvent:
    """带时间步的事件记录，用于世界模型的时间线。"""

    step: int
    event: AgentEvent


@dataclass
class EventTransitionStats:
    count: int = 0
    success_count: int = 0
    total_reward: float = 0.0


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
    transition_stats: Dict[tuple[str, str], EventTransitionStats] = field(default_factory=dict)
    concept_space: ConceptSpace = field(default_factory=ConceptSpace)
    concept_stats: Dict[ConceptId, ConceptStats] = field(default_factory=dict)
    timeline_limit: int = 500
    _timeline: List[TimedEvent] = field(default_factory=list, init=False, repr=False)
    _current_step: int = field(default=0, init=False, repr=False)
    last_brain_snapshot: Optional[BrainSnapshot] = field(default=None, init=False, repr=False)
    brain_snapshot_history: List[BrainSnapshot] = field(default_factory=list, init=False, repr=False)
    _last_context_key: Optional[str] = field(default=None, init=False, repr=False)
    _last_action_key: Optional[tuple[str, str]] = field(default=None, init=False, repr=False)

    def update(self, events: List[AgentEvent]) -> None:
        """将新事件写入历史，并更新工具统计。"""

        if not events:
            return

        for event in events:
            self.append_event(event)

    def advance_step(self) -> int:
        """
        增加一步时间刻度，返回当前 step。
        """

        self._current_step += 1
        return self._current_step

    @property
    def current_step(self) -> int:
        """返回当前世界模型的步数。"""

        return self._current_step

    def append_event(self, event: AgentEvent) -> None:
        """
        将事件追加到时间线，并做基础统计。
        """

        timed = TimedEvent(step=self._current_step, event=event)
        self._timeline.append(timed)
        if self.timeline_limit > 0 and len(self._timeline) > self.timeline_limit:
            overflow = len(self._timeline) - self.timeline_limit
            del self._timeline[0:overflow]

        event.meta.setdefault("step", self._current_step)
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
                "modalities": dict(stats.modalities),
                "last_seen_step": stats.last_seen_step,
            }

        return {
            "events": events_summary,
            "tools": tools_summary,
            "concepts": concept_summary,
            "brain": {
                "last_mode": getattr(self.last_brain_snapshot, "decision_hint", {}).get("mode")
                if self.last_brain_snapshot
                else None,
                "last_global_metrics": getattr(self.last_brain_snapshot, "global_metrics", {})
                if self.last_brain_snapshot
                else {},
            },
        }

    # 新增：概念相关观测接口 --------------------------------------------------------

    def observe_event(self, event: AgentEvent, concept: Optional[ConceptNode]) -> None:
        """
        在原有事件统计基础上，额外更新 concept_stats。
        """

        self.history.add(event)

        context_key = self._extract_context_key(event)
        action_key = self._extract_action_key(event)
        payload = event.payload if isinstance(event.payload, dict) else {}

        if self._last_context_key and action_key:
            transition_key = (self._last_context_key, action_key)
            stats = self.transition_stats.setdefault(transition_key, EventTransitionStats())
            stats.count += 1
            self._last_action_key = transition_key
        else:
            self._last_action_key = None

        kind = (
            event.kind.value
            if isinstance(event.kind, EventKind)
            else (event.kind or event.event_type)
        )
        if kind == EventKind.TOOL_RESULT.value:
            tool_name = str(payload.get("tool_name") or "unknown_tool")
            success = bool(payload.get("success"))
            stats = self.tool_stats.setdefault(tool_name, {"success": 0, "failure": 0})
            if success:
                stats["success"] += 1
            else:
                stats["failure"] += 1
            if action_key is None:
                action_key = tool_name

        if context_key and action_key:
            transition_key = (context_key, action_key)
            stats = self.transition_stats.setdefault(transition_key, EventTransitionStats())
            if stats.count == 0 and self._last_action_key != transition_key:
                stats.count = 1
            if payload and isinstance(payload.get("success"), bool) and payload.get("success"):
                stats.success_count += 1
            reward = self._extract_reward(event)
            if reward is not None:
                stats.total_reward += float(reward)

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
            modality = event.modality or "unknown"
            stats.modalities[modality] = int(stats.modalities.get(modality, 0)) + 1
            stats.last_seen_step = self._current_step

        self._last_context_key = context_key

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

    def recent_events(self, max_count: int = 20) -> List[TimedEvent]:
        """返回最近 max_count 条事件（倒序）。"""

        if max_count <= 0:
            return []
        return list(reversed(self._timeline[-max_count:]))

    def query_events(
        self,
        modality: Optional[str] = None,
        tag: Optional[str] = None,
        max_count: int = 50,
    ) -> List[TimedEvent]:
        """按模态/tag 过滤最近的事件。"""

        results: List[TimedEvent] = []
        for timed in reversed(self._timeline):
            if modality and timed.event.modality != modality:
                continue
            if tag and tag not in timed.event.tags:
                continue
            results.append(timed)
            if len(results) >= max_count:
                break
        return results

    def events_between(self, start_step: int, end_step: int) -> List[TimedEvent]:
        """返回指定步数区间内的事件。"""

        return [
            t for t in self._timeline if start_step <= t.step <= end_step
        ]

    def top_concepts(self, top_k: int = 10) -> List[tuple[ConceptNode, ConceptStats]]:
        """按出现次数排序的 top_k 概念。"""

        ordered = sorted(
            self.concept_stats.items(),
            key=lambda item: item[1].count,
            reverse=True,
        )
        concepts: List[tuple[ConceptNode, ConceptStats]] = []
        for cid, stats in ordered[:top_k]:
            node = next(
                (c for c in self.concept_space.all_concepts() if str(c.id) == str(cid)),
                None,
            )
            if node is not None:
                concepts.append((node, stats))
        return concepts

    def concepts_by_modality(
        self, modality: str, top_k: int = 10
    ) -> List[tuple[ConceptNode, ConceptStats]]:
        """在指定模态中最常见的概念。"""

        filtered = [
            (cid, stats)
            for cid, stats in self.concept_stats.items()
            if stats.modalities.get(modality)
        ]
        ordered = sorted(filtered, key=lambda item: item[1].modalities.get(modality, 0), reverse=True)
        results: List[tuple[ConceptNode, ConceptStats]] = []
        for cid, stats in ordered[:top_k]:
            node = next(
                (c for c in self.concept_space.all_concepts() if str(c.id) == str(cid)),
                None,
            )
            if node is not None:
                results.append((node, stats))
        return results

    def update_brain_snapshot(self, snapshot: BrainSnapshot, max_history: int = 5) -> None:
        """记录最新的 BrainSnapshot，为驱动力与对话提供参考。"""

        self.last_brain_snapshot = snapshot
        self.brain_snapshot_history.append(snapshot)
        if max_history > 0 and len(self.brain_snapshot_history) > max_history:
            overflow = len(self.brain_snapshot_history) - max_history
            del self.brain_snapshot_history[0:overflow]

    # 因果/预测相关的轻量接口 -----------------------------------------------------

    def _extract_context_key(self, event: AgentEvent) -> str:
        meta = event.meta if isinstance(event.meta, dict) else {}
        payload = event.payload if isinstance(event.payload, dict) else {}
        for key in ("scenario_id", "scene_id", "context_id"):
            if key in meta:
                return str(meta[key])
            if key in payload:
                return str(payload[key])
        return str(event.event_type or event.kind or "unknown")

    def _extract_action_key(self, event: AgentEvent) -> str | None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        tool = payload.get("tool_name")
        if tool:
            return str(tool)
        if event.event_type == EventKind.DIALOGUE.value:
            return "dialogue"
        if event.event_type == EventKind.TASK.value:
            return str(payload.get("task_type") or "task")
        return None

    def _extract_reward(self, event: AgentEvent) -> float | None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        meta = event.meta if isinstance(event.meta, dict) else {}
        for key in ("reward", "score"):
            val = payload.get(key, meta.get(key)) if isinstance(payload, dict) else meta.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        success = payload.get("success") if isinstance(payload, dict) else None
        if isinstance(success, bool):
            return 1.0 if success else -0.3
        return None

    def predict_success_prob(self, scenario_id: str, action_key: str) -> float:
        """
        使用 transition_stats 中 (scenario_id, action_key) 的历史统计，
        返回简单成功概率估计（成功次数 / 总次数），若数据不足则返回默认值。
        """

        key = (str(scenario_id), str(action_key))
        stats = self.transition_stats.get(key)
        if not stats or stats.count <= 0:
            return 0.5
        return stats.success_count / max(1, stats.count)
