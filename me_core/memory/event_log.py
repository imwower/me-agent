from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional

from me_core.types import AgentEvent, EventKind, EventSource

logger = logging.getLogger(__name__)


FilterFn = Callable[[AgentEvent], bool]


@dataclass
class EventLog:
    """用于存储结构化事件的简单日志。

    与 EventStream 的区别：
        - EventStream 更像“在线队列”，用于事件即时分发；
        - EventLog 更偏向“历史记录”，适合学习/工具发现等离线分析；
        - 当前实现使用同样的 AgentEvent 结构，便于互操作。
    """

    max_events: int = 5000
    _events: List[AgentEvent] = field(default_factory=list, init=False)

    def append(self, event: AgentEvent) -> None:
        """追加一条事件。"""

        self._events.append(event)
        if len(self._events) > self.max_events:
            overflow = len(self._events) - self.max_events
            del self._events[0:overflow]
        logger.info("EventLog.append: 当前事件总数=%d", len(self._events))

    def extend(self, events: Iterable[AgentEvent]) -> None:
        """批量追加事件。"""

        for e in events:
            self.append(e)

    def tail(self, limit: int) -> List[AgentEvent]:
        """返回最近 limit 条事件。"""

        if limit <= 0:
            return []
        return self._events[-limit:]

    def filter(
        self,
        *,
        kind: Optional[str] = None,
        source: Optional[str] = None,
        predicate: Optional[FilterFn] = None,
    ) -> List[AgentEvent]:
        """按条件筛选事件。

        参数：
            kind: 事件类型（event_type 或 EventKind 值）；
            source: 事件来源（EventSource 值）；
            predicate: 自定义过滤函数。
        """

        result: List[AgentEvent] = []
        for e in self._events:
            if kind is not None:
                if isinstance(kind, EventKind):
                    kind_value = kind.value
                else:
                    kind_value = str(kind)
                event_kind = (
                    e.kind.value
                    if isinstance(e.kind, EventKind)
                    else (e.kind or e.event_type)
                )
                if event_kind != kind_value:
                    continue

            if source is not None:
                if isinstance(source, EventSource):
                    source_value = source.value
                else:
                    source_value = str(source)
                if (e.source or "") != source_value:
                    continue

            if predicate is not None and not predicate(e):
                continue

            result.append(e)

        logger.info(
            "EventLog.filter: kind=%r, source=%r, 命中=%d 条",
            kind,
            source,
            len(result),
        )
        return result

