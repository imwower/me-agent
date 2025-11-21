from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

from me_core.types import AgentEvent

from .self_state import SelfState
from .self_summarizer import summarize_self
from .self_updater import update_from_event

if TYPE_CHECKING:
    from me_core.world_model.base import SimpleWorldModel

class BaseSelfModel(ABC):
    """自我模型基类。

    设计意图：
        - 封装 SelfState 的读写与更新逻辑；
        - 为 Agent 提供统一接口，而不关心内部如何表示自我状态；
        - 便于未来替换为更复杂的自我建模实现。
    """

    @abstractmethod
    def update_from_events(self, events: List[AgentEvent]) -> None:
        """根据一批事件更新自我状态。"""

    @abstractmethod
    def describe(self) -> str:
        """返回当前自我状态的一段中文自述。"""

    @abstractmethod
    def get_state(self) -> SelfState:
        """获取底层自我状态对象，便于外部检查或持久化。"""


class SimpleSelfModel(BaseSelfModel):
    """基于 SelfState 的简易自我模型实现。

    - 使用 update_from_event 逐条吸收事件；
    - 使用 summarize_self 生成自我描述；
    - 作为一个轻量包装层，便于在 Agent 之间复用相同的自我建模逻辑。
    """

    def __init__(self, state: SelfState | None = None) -> None:
        self._state: SelfState = state or SelfState()

    def update_from_events(self, events: List[AgentEvent]) -> None:
        """按顺序依次用事件驱动自我状态更新。"""

        for e in events:
            self.observe_event(e)

    def observe_event(self, event: AgentEvent) -> None:
        """根据单个事件更新自我模型。"""

        self._state = update_from_event(self._state, event)

    def describe(self) -> str:
        """将 summarize_self 的三个字段拼接为一段短文本。"""

        summary = summarize_self(self._state)
        parts = [
            summary.get("who_am_i", "").strip(),
            summary.get("what_can_i_do", "").strip(),
            summary.get("what_do_i_need", "").strip(),
        ]
        text = " ".join(p for p in parts if p)
        return text or "我目前还在初始化自己的自我模型。"

    def describe_self(self, world_model: Optional["SimpleWorldModel"] = None) -> str:
        """
        用简单中文描述当前“我”的状态：
        - 见过哪些模态（例如：文本/图像）
        - 拥有哪些能力标签（如时间查询、多模态 dummy 对齐）
        - （可选）最近常见概念（如果提供了 world_model）
        """

        segments: List[str] = []
        if self._state.seen_modalities:
            mods = "、".join(sorted(self._state.seen_modalities))
            segments.append(f"我目前见过的模态包括：{mods}")

        if self._state.capability_tags:
            tags = "、".join(sorted(self._state.capability_tags))
            segments.append(f"已具备的能力标签：{tags}")

        if world_model is not None:
            concepts = getattr(world_model, "recent_concepts", lambda top_k=3: [])(top_k=3)
            if concepts:
                names = "、".join(c.name for c in concepts[:3])
                segments.append(f"最近对齐较多的概念有：{names}")

        if not segments:
            return self.describe()

        return "；".join(segments)

    def get_state(self) -> SelfState:
        """返回当前自我状态。"""

        return self._state
