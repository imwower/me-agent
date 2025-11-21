from __future__ import annotations

from pathlib import Path
from typing import Any, List

from .base import BasePerception
from ..types import AgentEvent, AudioRef, EventKind


class AudioPerception(BasePerception):
    """
    音频感知桩：R0 仅包装路径为 AudioRef，不做真实解析。
    """

    def __init__(self, default_source: str = "environment") -> None:
        self.default_source = default_source

    def perceive(self, raw_input: Any, **kwargs) -> List[AgentEvent]:
        source = kwargs.get("source") or self.default_source

        if isinstance(raw_input, AudioRef):
            audio_ref = raw_input
        elif isinstance(raw_input, str):
            audio_ref = AudioRef(path=str(Path(raw_input)))
        else:
            raise TypeError("AudioPerception 目前仅支持 str 或 AudioRef 类型输入。")

        path_obj = Path(audio_ref.path)
        exists = path_obj.exists()
        payload = {
            "audio_ref": audio_ref,
            "path": audio_ref.path,
            "exists": exists,
        }
        if not exists:
            payload["error"] = "file_not_found"

        event = AgentEvent.now(
            event_type=EventKind.PERCEPTION.value,
            payload=payload,
            source=source,
            kind=EventKind.PERCEPTION,
        )
        event.modality = "audio"
        event.tags.update({"audio"})
        if not exists:
            event.tags.add("missing")
        return [event]

