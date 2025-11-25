from __future__ import annotations

from pathlib import Path
from typing import Any, List

from .base import BasePerception
from ..types import AgentEvent, EventKind, VideoRef


class VideoPerception(BasePerception):
    """
    视频感知桩：仅记录路径与基础元数据，不做帧解析。
    """

    def __init__(self, default_source: str = "environment") -> None:
        self.default_source = default_source

    def perceive(self, raw_input: Any, **kwargs) -> List[AgentEvent]:
        source = kwargs.get("source") or self.default_source

        if isinstance(raw_input, VideoRef):
            video_ref = raw_input
        elif isinstance(raw_input, str):
            video_ref = VideoRef(path=str(Path(raw_input)))
        elif isinstance(raw_input, dict):
            path = raw_input.get("path") or ""
            meta = raw_input.get("meta") or {}
            video_ref = VideoRef(path=str(path), meta=meta)
        else:
            raise TypeError("VideoPerception 目前仅支持 str/VideoRef/dict 输入。")

        path_obj = Path(video_ref.path)
        exists = path_obj.exists()
        payload = {
            "video_path": video_ref.path,
            "video_ref": video_ref,
            "exists": exists,
            "meta": dict(video_ref.meta),
        }
        if not exists:
            payload["error"] = "file_not_found"

        event = AgentEvent.now(
            event_type=EventKind.PERCEPTION.value,
            payload=payload,
            source=source,
            kind=EventKind.PERCEPTION,
        )
        event.modality = "video"
        event.tags.update({"video"})
        if not exists:
            event.tags.add("missing")
        return [event]
