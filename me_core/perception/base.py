from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from me_core.types import AgentEvent, EventKind, EventSource, MultiModalInput

from .processor import encode_to_event


class BasePerception(ABC):
    """感知模块基类。

    设计意图：
        - 将“原始输入”转换为统一的 AgentEvent 结构；
        - 为后续扩展多模态感知（文本、图像、音频等）预留接口；
        - 不在此层处理复杂语义，只做“规整和打包”。
    """

    @abstractmethod
    def perceive(self, raw_input: Any) -> AgentEvent:
        """将原始输入转为 AgentEvent。

        不同实现可以假定 raw_input 的类型不同（如 str / dict / MultiModalInput），
        但返回值一律为统一的事件结构，方便写入事件流。
        """


class TextPerception(BasePerception):
    """最简文本感知实现。

    当前仅支持字符串输入，将其包装为单模态的 MultiModalInput，
    再借助 encode_to_event 统一转换为 AgentEvent。
    """

    def __init__(self, default_source: str = EventSource.HUMAN.value) -> None:
        self.default_source = default_source

    def perceive(self, raw_input: Any) -> AgentEvent:
        """将文本输入转换为感知事件。

        事件 payload 示例：
            kind: "perception"
            source: "human"
            raw.text: 原始文本
        """

        if not isinstance(raw_input, str):
            raise TypeError("TextPerception 目前仅支持 str 类型输入。")

        mm_input = MultiModalInput(text=raw_input)
        event = encode_to_event(mm_input, source=self.default_source)
        # 同步补充 AgentEvent 自身的来源与类型信息，方便后续过滤
        event.source = self.default_source
        event.kind = EventKind.PERCEPTION
        return event

