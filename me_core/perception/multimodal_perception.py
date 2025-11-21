from __future__ import annotations

from typing import Any, List

from .base import BasePerception
from .processor import encode_to_event
from .text_perception import TextPerception
from ..types import AgentEvent, EventKind, MultiModalInput


class MultiModalPerception(BasePerception):
    """简单多模态感知实现（兼容旧接口）。"""

    def __init__(self, default_source: str = "human") -> None:
        self.default_source = default_source
        # 保持单事件输出，关闭分句
        self.text_perception = TextPerception(split_sentences=False, default_source=default_source)

    def perceive(self, raw_input: Any, **kwargs) -> List[AgentEvent]:
        source = kwargs.get("source") or self.default_source

        if isinstance(raw_input, MultiModalInput):
            event = encode_to_event(raw_input, source=source)
            event.source = source
            event.kind = EventKind.PERCEPTION

            modalities: List[str] = []
            if raw_input.text is not None:
                modalities.append("text")
            if raw_input.image_meta is not None:
                modalities.append("image")
            if raw_input.audio_meta is not None:
                modalities.append("audio")
            if raw_input.video_meta is not None:
                modalities.append("video")

            if len(modalities) == 1:
                event.modality = modalities[0]
            elif modalities:
                event.modality = "mixed"

            if isinstance(event.tags, set):
                event.tags.update(modalities or {"mixed"})

            return [event]

        if isinstance(raw_input, (str, list)):
            return self.text_perception.perceive(raw_input, source=source)

        raise TypeError("MultiModalPerception 目前仅支持 str、list[str] 或 MultiModalInput 类型输入。")

