from __future__ import annotations

import re
from typing import Any, List

from .base import BasePerception
from ..types import AgentEvent, EventKind, EventSource


class TextPerception(BasePerception):
    """
    文本感知：将字符串拆分为一个或多个文本事件。
    """

    def __init__(self, split_sentences: bool = True, default_source: str = EventSource.HUMAN.value) -> None:
        self.split_sentences = split_sentences
        self.default_source = default_source

    def _split_text(self, text: str) -> List[str]:
        """按行与简单标点拆分文本。"""

        if not self.split_sentences:
            return [text.strip()] if text.strip() else []

        segments: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[。！？!?]", line)
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    segments.append(cleaned)
        if not segments and text.strip():
            segments.append(text.strip())
        return segments

    def perceive(self, raw_input: Any, **kwargs) -> List[AgentEvent]:
        # 若 raw_input 是 list[str]，逐条处理；若为 str，则按配置拆分
        source = kwargs.get("source") or self.default_source
        texts: List[str]
        if isinstance(raw_input, list):
            if not all(isinstance(item, str) for item in raw_input):
                raise TypeError("TextPerception 仅支持 str 或 list[str] 类型输入。")
            texts = [t for t in raw_input if isinstance(t, str)]
        elif isinstance(raw_input, str):
            texts = [raw_input]
        else:
            raise TypeError("TextPerception 仅支持 str 或 list[str] 类型输入。")

        events: List[AgentEvent] = []
        for text in texts:
            for segment in self._split_text(text):
                payload = {"text": segment, "raw": {"text": segment}}
                event = AgentEvent.now(
                    event_type=EventKind.PERCEPTION.value,
                    payload=payload,
                    source=source,
                    kind=EventKind.PERCEPTION,
                )
                event.modality = "text"
                event.tags.update({"text"})
                events.append(event)
        return events
