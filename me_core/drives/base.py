from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Literal

from me_core.alignment.concepts import ConceptId
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
    """表示当前时刻的高层意图。"""

    kind: Literal[
        "reply",
        "call_tool",
        "stay_silent",
        "curiosity",
        "reflect_self",
        "inspect_world",
    ]
    priority: int = 0
    target_concept_id: ConceptId | None = None
    preferred_modality: str | None = None
    message: str | None = None  # 意图的中文说明，便于日志和调试
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
    curiosity_min_count: int = 2
    reflect_gap_steps: int = 5
    enable_curiosity: bool = True
    enable_reflection: bool = True
    policy_config: Any = None  # 占位，用于注入 AgentPolicy

    def decide_intent(
        self,
        self_model: "BaseSelfModel",
        world_model: "BaseWorldModel",
        recent_events: List[AgentEvent],
    ) -> Intent:
        """根据最近事件与世界/自我状态给出当前意图。"""

        now = datetime.now(timezone.utc)
        candidates: list[Intent] = []
        state = self_model.get_state()
        brain_mode = getattr(state, "last_brain_mode", "unknown")
        brain_conf = float(getattr(state, "last_brain_confidence", 0.0) or 0.0)

        def add(intent: Intent) -> None:
            candidates.append(intent)

        # 默认保持安静
        add(Intent(kind="stay_silent", priority=0, explanation="默认保持安静。", extra={"reason": "baseline"}))

        # 1. 根据最新事件生成基础 reply/call_tool/stay_silent 候选
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
            user_text = ""
            payload = last_human_perception.payload or {}
            raw = payload.get("raw")
            if isinstance(raw, dict) and isinstance(raw.get("text"), str):
                user_text = raw["text"]

            priority = 6
            explanation = "最近收到一条来自用户的输入，需要做出回应。"
            if any(keyword in user_text for keyword in ("时间", "time", "几点")):
                explanation = "用户在询问时间，打算调用时间相关工具来回答。"
                add(
                    Intent(
                        kind="call_tool",
                        target_tool="time",
                        priority=8,
                        explanation=explanation,
                        message="帮用户确认时间后再回复",
                        extra={"reason": "user_ask_time", "tool_args": {}},
                    )
                )
            else:
                add(
                    Intent(
                        kind="reply",
                        priority=priority,
                        explanation=explanation,
                        message="回应最新的用户输入",
                        extra={"reason": "recent_human_input"},
                    )
                )

        # 2. 根据 world + concept_stats 生成 curiosity 候选
        concept_stats = getattr(world_model, "concept_stats", {}) or {}
        concept_space = getattr(world_model, "concept_space", None)
        interesting: Optional[tuple[ConceptId, Any]] = None
        for cid, stats in concept_stats.items():
            modalities = getattr(stats, "modalities", {}) or {}
            if isinstance(modalities, dict) and len(modalities.keys()) == 1:
                count = getattr(stats, "count", 0)
                threshold = self.curiosity_min_count
                if self.policy_config and getattr(self.policy_config, "curiosity", None):
                    try:
                        threshold = int(self.policy_config.curiosity.min_concept_count)
                    except Exception:
                        pass
                if count >= threshold:
                    interesting = (cid, stats)
                    break
            elif isinstance(modalities, (list, set)) and len(modalities) == 1:
                count = getattr(stats, "count", 0)
                if count >= self.curiosity_min_count:
                    interesting = (cid, stats)
                    break

        if self.enable_curiosity and interesting is not None:
            cid, stats = interesting
            name: str | None = None
            if concept_space is not None and hasattr(concept_space, "all_concepts"):
                for concept in concept_space.all_concepts():  # type: ignore[attr-defined]
                    if str(concept.id) == str(cid):
                        name = concept.name
                        break
            target_name = name or str(cid)
            preferred_modality = None
            if isinstance(getattr(stats, "modalities", None), dict):
                only_modality = next(iter(stats.modalities.keys()))
                preferred_modality = "image" if only_modality == "text" else None
            add(
                Intent(
                    kind="curiosity",
                    priority=4,
                    target_concept_id=cid,
                    preferred_modality=preferred_modality,
                    message=f"概念「{target_name}」需要更多模态信息",
                    explanation=f"概念「{target_name}」已多次出现但只有单一模态，希望获得更多相关信息。",
                    extra={
                        "reason": "curiosity_multimodal",
                        "concept_id": cid,
                        "concept_name": target_name,
                    },
                )
            )
        elif self.enable_curiosity and brain_mode == "explore" and brain_conf > 0.5:
            add(
                Intent(
                    kind="curiosity",
                    priority=4,
                    explanation="脑状态建议探索，先寻找更多信息。",
                    message="跟进不确定点，收集信息",
                    extra={"reason": "brain_mode_explore"},
                )
            )

        # 3. 根据 self_state 决定是否生成 reflect_self / inspect_world 候选
        current_step = getattr(world_model, "_current_step", 0)
        if self.enable_reflection and current_step - state.last_step >= self.reflect_gap_steps:
            add(
                Intent(
                    kind="reflect_self",
                    priority=3,
                    explanation="有一段时间没有总结自我状态，准备自述当前能力与关注。",
                    message="做一次自我描述",
                    extra={"reason": "self_reflection_gap"},
                )
            )

        if getattr(world_model, "_timeline", []):
            add(
                Intent(
                    kind="inspect_world",
                    priority=2,
                    explanation="回顾一下最近的世界事件和概念。",
                    message="整理最新世界观察",
                    extra={"reason": "inspect_world_state"},
                )
            )

        # 4. 根据时间怠速情况加入保持沉默或自省
        if recent_events:
            last_any = recent_events[-1]
            idle_seconds = (now - last_any.timestamp).total_seconds()
            if idle_seconds >= self.idle_threshold_seconds:
                add(
                    Intent(
                        kind="stay_silent",
                        priority=1,
                        explanation="长时间没有新事件，先保持安静观察。",
                        message="等待新的信号",
                        extra={"reason": "idle_timeout"},
                    )
                )

        if not candidates:
            return Intent(kind="stay_silent", priority=0, explanation="暂无候选意图。", extra={"reason": "empty"})

        # 5. 若脑状态给出偏好，对候选意图做轻微调权
        if brain_mode == "explore" and brain_conf > 0.5:
            for cand in candidates:
                if cand.kind in ("curiosity", "inspect_world", "reflect_self"):
                    cand.priority += 1
        elif brain_mode == "exploit" and brain_conf > 0.5:
            for cand in candidates:
                if cand.kind in ("reply", "call_tool"):
                    cand.priority += 1
        elif brain_mode == "chaotic":
            for cand in candidates:
                if cand.kind == "stay_silent":
                    cand.priority += 1

        # 6. 按 priority 和简单的 tie-break 规则选出一个 Intent 返回
        best = max(
            enumerate(candidates),
            key=lambda pair: (pair[1].priority, pair[0]),
        )[1]
        return best
