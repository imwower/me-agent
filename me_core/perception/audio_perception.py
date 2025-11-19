from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from me_core.types import AgentEvent, AudioRef, EventKind, EventSource, MultiModalInput

from .base import BasePerception
from .processor import encode_to_event


class AudioPerception(BasePerception):
    """音频感知桩实现。

    当前仅记录文件路径与基础元信息，不做波形或特征提取。
    """

    def __init__(self, default_source: str = EventSource.ENVIRONMENT.value) -> None:
        self.default_source = default_source

    def perceive(self, raw_input: Any) -> AgentEvent:
        if isinstance(raw_input, AudioRef):
            audio_ref = raw_input
        elif isinstance(raw_input, str):
            path = Path(raw_input)
            audio_ref = AudioRef(path=str(path))
        else:
            raise TypeError("AudioPerception 目前仅支持 str 或 AudioRef 类型输入。")

        mm_input = MultiModalInput(audio_meta=asdict(audio_ref))
        event = encode_to_event(mm_input, source=self.default_source)
        event.source = self.default_source
        event.kind = EventKind.PERCEPTION
        event.modality = "audio"
        if isinstance(event.tags, set):
            event.tags.update({"audio"})
        return event

