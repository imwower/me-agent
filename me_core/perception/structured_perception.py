from __future__ import annotations

import json
from typing import Any, Dict, List

from .base import BasePerception
from ..types import AgentEvent, EventKind


class StructuredPerception(BasePerception):
    """
    结构化数据感知：
    - 支持 dict（直接视为结构化数据）
    - 支持 JSON 字符串（可解析为 dict）
    """

    def perceive(self, raw_input: Any, **kwargs) -> List[AgentEvent]:
        source = kwargs.get("source") or "environment"

        data: Dict[str, Any] | None = None
        if isinstance(raw_input, dict):
            data = raw_input
        elif isinstance(raw_input, str):
            try:
                obj = json.loads(raw_input)
                if isinstance(obj, dict):
                    data = obj
            except Exception:
                data = None

        if data is None:
            raise TypeError("StructuredPerception 仅支持 dict 或可解析为 dict 的 JSON 字符串。")

        event = AgentEvent.now(
            event_type=EventKind.PERCEPTION.value,
            payload={"kind": "structured", "data": data},
            source=source,
            kind=EventKind.PERCEPTION,
        )
        event.modality = "structured"
        event.tags.update({"structured"})
        return [event]
