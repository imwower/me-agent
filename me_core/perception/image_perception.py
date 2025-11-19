from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from me_core.types import AgentEvent, EventKind, EventSource, ImageRef, MultiModalInput

from .base import BasePerception
from .processor import encode_to_event


class ImagePerception(BasePerception):
    """图像感知实现（轻量版）。

    职责：
        - 接收本地路径或 ImageRef；
        - 检查文件是否存在（若为本地路径）；
        - 将基础信息封装为 MultiModalInput.image_meta，并生成 AgentEvent。
    """

    def __init__(self, default_source: str = EventSource.ENVIRONMENT.value) -> None:
        self.default_source = default_source

    def perceive(self, raw_input: Any) -> AgentEvent:
        if isinstance(raw_input, ImageRef):
            img_ref = raw_input
        elif isinstance(raw_input, str):
            path = Path(raw_input)
            # 若路径不存在，仍然记录原始字符串，交由上层决定是否报错
            img_ref = ImageRef(path=str(path))
        else:
            raise TypeError("ImagePerception 目前仅支持 str 或 ImageRef 类型输入。")

        mm_input = MultiModalInput(image_meta=asdict(img_ref))
        event = encode_to_event(mm_input, source=self.default_source)
        event.source = self.default_source
        event.kind = EventKind.PERCEPTION
        event.modality = "image"
        if isinstance(event.tags, set):
            event.tags.update({"image"})
        return event

