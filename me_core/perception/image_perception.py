from __future__ import annotations

from pathlib import Path
from typing import Any, List

from .base import BasePerception
from ..types import AgentEvent, EventKind, ImageRef


class ImagePerception(BasePerception):
    """
    图片感知：目前 R0 版仅记录路径和简单 meta，不做真实像素解析。
    """

    def __init__(self, default_source: str = "environment") -> None:
        self.default_source = default_source

    def perceive(self, raw_input: Any, **kwargs) -> List[AgentEvent]:
        source = kwargs.get("source") or self.default_source

        if isinstance(raw_input, ImageRef):
            image_ref = raw_input
        elif isinstance(raw_input, str):
            image_ref = ImageRef(path=str(Path(raw_input)))
        else:
            raise TypeError("ImagePerception 目前仅支持 str 或 ImageRef 类型输入。")

        path_obj = Path(image_ref.path)
        exists = path_obj.exists()

        payload = {
            "image_ref": image_ref,
            "path": image_ref.path,
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
        event.modality = "image"
        event.tags.update({"image"})
        if not exists:
            event.tags.add("missing")

        return [event]

