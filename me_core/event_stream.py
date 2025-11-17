from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, Iterable, Iterator, List, Optional

from .types import AgentEvent, EventKind, EventSource


@dataclass
class EventStream:
    """简单的内存事件流。

    设计意图：
        - 在原型阶段，仅使用内存中的 deque 存放事件队列；
        - 为上层 Agent 提供统一的 append / 迭代 / 查询接口；
        - 日后可替换为基于消息队列或数据库的实现，而不影响调用方。
    """

    max_events: int = 1000
    _events: Deque[AgentEvent] = field(default_factory=deque, init=False)

    def append_event(self, event: AgentEvent) -> None:
        """追加一条事件到流中。

        若超出 max_events，则自动丢弃最旧的事件。
        """

        self._events.append(event)
        while len(self._events) > self.max_events:
            self._events.popleft()

    def iter_events(
        self,
        *,
        source: Optional[str] = None,
        kind: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Iterator[AgentEvent]:
        """按简单条件过滤遍历事件。

        参数：
            source: 事件来源过滤（字符串或 EventSource.value）。
            kind: 事件类型过滤（event_type 或 EventKind.value）。
            since: 起始时间（含），若为 None 则不限制。
            until: 结束时间（含），若为 None 则不限制。
        """

        for event in self._events:
            if since is not None and event.timestamp < since:
                continue
            if until is not None and event.timestamp > until:
                continue

            if source is not None:
                src = event.source or ""
                if isinstance(source, EventSource):
                    source_value = source.value
                else:
                    source_value = str(source)
                if src != source_value:
                    continue

            if kind is not None:
                if isinstance(kind, EventKind):
                    kind_value = kind.value
                else:
                    kind_value = str(kind)
                event_kind = (
                    event.kind.value
                    if isinstance(event.kind, EventKind)
                    else (event.kind or event.event_type)
                )
                if event_kind != kind_value:
                    continue

            yield event

    def last_event(
        self,
        kind: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Optional[AgentEvent]:
        """返回满足条件的最后一条事件，若不存在则返回 None。"""

        # 由于内部结构是 deque，倒序遍历可以较快找到最近匹配项
        for event in reversed(self._events):  # type: ignore[arg-type]
            if kind is not None:
                if isinstance(kind, EventKind):
                    kind_value = kind.value
                else:
                    kind_value = str(kind)
                event_kind = (
                    event.kind.value
                    if isinstance(event.kind, EventKind)
                    else (event.kind or event.event_type)
                )
                if event_kind != kind_value:
                    continue

            if source is not None:
                src = event.source or ""
                if isinstance(source, EventSource):
                    source_value = source.value
                else:
                    source_value = str(source)
                if src != source_value:
                    continue

            return event
        return None

    def log_event(self, event: AgentEvent) -> None:
        """将事件以统一的中文格式打印到 stdout。

        目前直接使用 print，后续可以替换为 logging 或结构化日志。
        """

        # 这里使用事件自身的 pretty() 方法，保证日志格式在一个地方维护
        print(event.pretty())  # noqa: T201

    def to_list(self) -> List[AgentEvent]:
        """返回当前事件流的浅拷贝列表。

        该方法主要用于测试或需要一次性遍历全部事件的场景。
        """

        return list(self._events)


@dataclass
class EventHistory:
    """事件历史存储与摘要工具。

    设计意图：
        - 为 world_model / self_model 提供一个轻量的“最近 N 条事件”视图；
        - 支持按类型与来源汇总统计，为后续策略学习提供输入；
        - 当前仅在内存中工作，不做持久化。
    """

    max_events: int = 500
    _events: Deque[AgentEvent] = field(default_factory=deque, init=False)

    def add(self, event: AgentEvent) -> None:
        """追加一条事件并维护长度上限。"""

        self._events.append(event)
        while len(self._events) > self.max_events:
            self._events.popleft()

    def extend(self, events: Iterable[AgentEvent]) -> None:
        """批量追加事件。"""

        for e in events:
            self.add(e)

    def recent(self, limit: Optional[int] = None) -> List[AgentEvent]:
        """返回最近若干条事件。"""

        if limit is None or limit >= len(self._events):
            return list(self._events)
        # deque 不支持直接切片，这里先转换为列表
        items = list(self._events)
        return items[-limit:]

    def summarize(self) -> Dict[str, object]:
        """生成一个简要摘要，供 world_model / self_model 使用。

        摘要内容包括：
            - total: 总事件数
            - by_kind: 各类型事件计数
            - by_source: 各来源事件计数
            - first_timestamp / last_timestamp: 时间范围
        """

        total = len(self._events)
        by_kind: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        first_ts: Optional[datetime] = None
        last_ts: Optional[datetime] = None

        for e in self._events:
            # 统计类型
            k = (
                e.kind.value
                if isinstance(e.kind, EventKind)
                else (e.kind or e.event_type)
            )
            by_kind[k] = by_kind.get(k, 0) + 1

            # 统计来源
            s = e.source or "unknown"
            by_source[s] = by_source.get(s, 0) + 1

            # 更新时间范围
            if first_ts is None or e.timestamp < first_ts:
                first_ts = e.timestamp
            if last_ts is None or e.timestamp > last_ts:
                last_ts = e.timestamp

        return {
            "total": total,
            "by_kind": by_kind,
            "by_source": by_source,
            "first_timestamp": first_ts.isoformat() if first_ts else None,
            "last_timestamp": last_ts.isoformat() if last_ts else None,
        }

