from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

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

        # 为后续多模态感知预留分句扩展位：当前版本仍将整段文本视作单个事件，
        # 但可以在 payload 中记录原始文本，后续由上层模块按需切分。
        mm_input = MultiModalInput(text=raw_input)
        event = encode_to_event(mm_input, source=self.default_source)
        # 同步补充 AgentEvent 自身的来源与类型信息，方便后续过滤
        event.source = self.default_source
        event.kind = EventKind.PERCEPTION
        event.modality = "text"
        if isinstance(event.tags, set):
            event.tags.update({"text", "user_input"})
        return event


class MultiModalPerception(BasePerception):
    """简单多模态感知实现。

    约定：
        - raw_input 为 str 时，退化为纯文本感知；
        - raw_input 为 MultiModalInput 时，直接走多模态编码；
        - 后续可扩展为支持 ImageRef / AudioRef 等结构。
    """

    def __init__(self, default_source: str = EventSource.HUMAN.value) -> None:
        self.default_source = default_source

    def perceive(self, raw_input: Any) -> AgentEvent:
        if isinstance(raw_input, MultiModalInput):
            mm_input = raw_input
        elif isinstance(raw_input, str):
            mm_input = MultiModalInput(text=raw_input)
        else:
            raise TypeError(
                "MultiModalPerception 目前仅支持 str 或 MultiModalInput 类型输入。"
            )

        event = encode_to_event(mm_input, source=self.default_source)
        event.source = self.default_source
        event.kind = EventKind.PERCEPTION

        modalities: List[str] = []
        if mm_input.text is not None:
            modalities.append("text")
        if mm_input.image_meta is not None:
            modalities.append("image")
        if mm_input.audio_meta is not None:
            modalities.append("audio")
        if mm_input.video_meta is not None:
            modalities.append("video")

        if len(modalities) == 1:
            event.modality = modalities[0]
        elif modalities:
            event.modality = "mixed"

        if isinstance(event.tags, set):
            event.tags.update(modalities)

        return event

