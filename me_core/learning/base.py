from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING

from me_core.types import AgentEvent
from .policy_learner import PolicyLearner

if TYPE_CHECKING:  # 仅供类型检查使用，避免运行时循环依赖
    from me_core.drives.base import BaseDriveSystem
    from me_core.self_model.base import BaseSelfModel
    from me_core.tools.base import BaseTool
    from me_core.world_model.base import BaseWorldModel


@dataclass(slots=True)
class ToolUsageStats:
    call_count: int = 0
    success_count: int = 0


@dataclass(slots=True)
class IntentOutcomeStats:
    tried: int = 0
    succeeded: int = 0


class BaseLearner(ABC):
    """学习模块基类。

    设计意图：
        - 从事件流中“观察”到经验；
        - 在合适的时机更新 world_model / self_model / drives / tools；
        - 当前版本仅做轻量统计，为未来更复杂的学习算法预留接口。
    """

    @abstractmethod
    def observe(self, events: List[AgentEvent]) -> None:
        """观察一批新事件。"""

    @abstractmethod
    def update_models(
        self,
        world_model: "BaseWorldModel",
        self_model: "BaseSelfModel",
        drive_system: "BaseDriveSystem",
        tools: Dict[str, "BaseTool"],
    ) -> None:
        """基于当前累积的信息，对各子模型做一次增量更新。"""


@dataclass
class SimpleLearner(BaseLearner):
    """极简学习器实现。

    当前仅统计“观察到多少条事件”，不主动修改任何子模型结构，
    主要用于在 demo 中展示“学习模块”这一插拔点。
    """

    observed_event_count: int = 0
    concept_modalities: Dict[str, set[str]] = None  # type: ignore[assignment]
    concept_counts: Dict[str, int] = None  # type: ignore[assignment]
    tool_stats: Dict[str, ToolUsageStats] = field(default_factory=dict)
    intent_stats: Dict[str, IntentOutcomeStats] = field(default_factory=dict)
    policy_learner: PolicyLearner = field(default_factory=PolicyLearner)

    def observe(self, events: List[AgentEvent]) -> None:
        if self.concept_modalities is None:
            self.concept_modalities = {}
        if self.concept_counts is None:
            self.concept_counts = {}

        self.observed_event_count += len(events)
        for e in events:
            cid = e.meta.get("concept_id") if isinstance(e.meta, dict) else None
            if cid is None:
                continue
            cid_str = str(cid)
            self.concept_counts[cid_str] = int(self.concept_counts.get(cid_str, 0)) + 1
            if e.modality:
                mods = self.concept_modalities.setdefault(cid_str, set())
                mods.add(e.modality)

    def observe_tool_result(self, tool_name: str, success: bool) -> None:
        stats = self.tool_stats.setdefault(tool_name, ToolUsageStats())
        stats.call_count += 1
        if success:
            stats.success_count += 1

    def observe_intent_outcome(self, intent, success: bool) -> None:
        name = getattr(intent, "kind", "unknown_intent")
        stats = self.intent_stats.setdefault(name, IntentOutcomeStats())
        stats.tried += 1
        if success:
            stats.succeeded += 1
        if name == "curiosity":
            reward = 1.0 if success else -0.5
            self.policy_learner.record_outcome("curiosity.min_concept_count", reward, success)

    def update_models(
        self,
        world_model: "BaseWorldModel",
        self_model: "BaseSelfModel",
        drive_system: "BaseDriveSystem",
        tools: Dict[str, "BaseTool"],
    ) -> None:
        """当前版本不对模型做任何修改，仅作为扩展位。

        TODO:
            - 将 world_model 摘要与 self_model 状态作为输入；
            - 根据事件统计与工具表现，对驱动力或工具选择策略做轻微调整。
        """

        # 为了避免“未使用参数”告警，这里简单访问一次参数。
        _ = (world_model, self_model, drive_system, tools)
        policy = getattr(drive_system, "policy_config", None)
        if policy is None:
            return
        updates = self.policy_learner.propose_updates(policy)
        if updates:
            self.policy_learner.apply_updates(policy, updates)

    def get_tool_recommendations(self) -> List[tuple[str, ToolUsageStats]]:
        """
        返回按成功率/调用次数排序的工具列表。
        """

        items = list(self.tool_stats.items())
        items.sort(
            key=lambda item: (
                (item[1].success_count / item[1].call_count) if item[1].call_count else 0.0,
                item[1].call_count,
            ),
            reverse=True,
        )
        return items
