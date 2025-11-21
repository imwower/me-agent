from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from me_core.types import AgentEvent, EventKind, EventSource

if TYPE_CHECKING:  # 仅用于类型检查，避免运行时循环依赖
    from me_core.self_model.base import BaseSelfModel
    from me_core.world_model.base import BaseWorldModel


@dataclass(slots=True)
class Drive:
    """表示一种抽象的内在驱动力。

    当前实现仅作为语义占位，用于在日志和配置中描述“是什么驱动了当前意图”。
    真正的数值强度仍由 DriveVector 管理。
    """

    name: str
    description: str = ""


@dataclass(slots=True)
class Intent:
    """表示当前时刻的高层意图。

    字段示例：
        kind: "reply" / "call_tool" / "idle" / "reflect" 等；
        target_tool: 若需要调用工具，则指出工具名，例如 "time"；
        explanation: 一句中文解释，说明“我为什么有这个意图”；
        extra: 额外信息，例如工具参数、关联事件 id 等。
    """

    kind: str
    target_tool: Optional[str] = None
    explanation: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseDriveSystem(ABC):
    """驱动力决策系统基类。

    职责：
        - 结合 self_model + world_model + 最近事件，生成一个 Intent；
        - 不直接执行行动，只负责“我想 / 我要”的部分。
    """

    @abstractmethod
    def decide_intent(
        self,
        self_model: "BaseSelfModel",
        world_model: "BaseWorldModel",
        recent_events: List[AgentEvent],
    ) -> Intent:
        """基于当前内部/外部状态做一次意图决策。"""


@dataclass
class SimpleDriveSystem(BaseDriveSystem):
    """非常简化的驱动力系统实现。

    规则示意：
        - 若最近有来自人类的感知事件，则优先意图为“回复用户”；
        - 若长时间没有任何事件，则意图为“保持安静 / 自我反思”；
        - 否则保持 idle，不主动发起动作。
    """

    idle_threshold_seconds: float = 60.0

    def decide_intent(
        self,
        self_model: "BaseSelfModel",
        world_model: "BaseWorldModel",
        recent_events: List[AgentEvent],
    ) -> Intent:
        """根据最近事件与世界模型概要给出当前意图。"""

        _ = world_model  # 当前实现暂未直接使用世界模型，仅作为扩展位
        now = datetime.now(timezone.utc)

        # 1) 优先检查最近是否有来自“人类”的感知事件
        last_human_perception: Optional[AgentEvent] = None
        for e in reversed(recent_events):
            kind = (
                e.kind.value
                if isinstance(e.kind, EventKind)
                else (e.kind or e.event_type)
            )
            if kind != EventKind.PERCEPTION.value:
                continue
            src = e.source or ""
            if src in (
                EventSource.HUMAN.value,
                "cli_user_input",
                "cli_user",
            ):
                last_human_perception = e
                break

        if last_human_perception is not None:
            # 从感知事件中提取原始用户文本，辅助判断是否需要调用工具
            user_text = ""
            payload = last_human_perception.payload or {}
            raw = payload.get("raw")
            if isinstance(raw, dict):
                text_value = raw.get("text")
                if isinstance(text_value, str):
                    user_text = text_value

            # 简单规则：若用户提到“时间”等关键词，则意图为调用时间工具
            if any(keyword in user_text for keyword in ("时间", "time", "几点")):
                explanation = "用户在询问时间，打算调用时间相关工具来回答。"
                return Intent(
                    kind="call_tool",
                    target_tool="time",
                    explanation=explanation,
                    extra={"reason": "user_ask_time", "tool_args": {}},
                )

            explanation = "最近收到一条来自用户的输入，需要做出回应。"
            return Intent(
                kind="reply",
                target_tool=None,
                explanation=explanation,
                extra={"reason": "recent_human_input"},
            )

        # 1.5) 若最近未收到用户输入，但概念空间存在“单一模态、重复出现”的概念，
        #      触发好奇心，引导用户提供更多模态（例如图片或文字）。
        concept_stats = getattr(world_model, "concept_stats", {}) or {}
        concept_space = getattr(world_model, "concept_space", None)

        def _get_count(stats: Any) -> int:
            if hasattr(stats, "count"):
                return int(getattr(stats, "count", 0))
            if isinstance(stats, dict):
                return int(stats.get("count", 0))
            return 0

        def _get_modalities(stats: Any) -> set[str]:
            if hasattr(stats, "modalities"):
                mods = getattr(stats, "modalities", set()) or set()
                if isinstance(mods, dict):
                    return set(mods.keys())
                return set(mods)
            if isinstance(stats, dict):
                mods = stats.get("modalities") or {}
                if isinstance(mods, dict):
                    return set(mods.keys())
                if isinstance(mods, (list, set, tuple)):
                    return set(mods)
            return set()

        curious_target = None
        for cid, stats in concept_stats.items():
            modalities = _get_modalities(stats)
            if len(modalities) <= 1 and _get_count(stats) >= 2:
                curious_target = (cid, stats)
                break

        if curious_target is not None:
            cid, stats = curious_target
            name: str | None = None
            if concept_space is not None and hasattr(concept_space, "all_concepts"):
                for concept in concept_space.all_concepts():  # type: ignore[attr-defined]
                    if str(concept.id) == str(cid):
                        name = concept.name
                        break
            if name is None:
                name = str(cid)

            explanation = (
                f"概念「{name}」已多次出现但只有单一模态，希望获得更多相关信息。"
            )
            return Intent(
                kind="reply",
                target_tool=None,
                explanation=explanation,
                extra={
                    "reason": "curiosity_multimodal",
                    "concept_id": cid,
                    "concept_name": name,
                },
            )

        # 2) 没有任何事件：说明智能体还未真正启动，保持安静
        if not recent_events:
            return Intent(
                kind="idle",
                explanation="尚未感知到任何事件，先保持安静。",
                extra={"reason": "no_events"},
            )

        # 3) 长时间没有新事件：可以选择进行简短的自我反思（当前仅影响文案）
        last_any = recent_events[-1]
        idle_seconds = (now - last_any.timestamp).total_seconds()
        if idle_seconds >= self.idle_threshold_seconds:
            return Intent(
                kind="reflect",
                explanation="长时间没有新事件，进行简单自我反思。",
                extra={"reason": "idle_timeout"},
            )

        # 4) 默认保持安静但保持关注
        return Intent(
            kind="idle",
            explanation="最近没有需要立即回应的事件，暂时保持安静。",
            extra={"reason": "nothing_urgent"},
        )
