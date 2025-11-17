import unittest
from datetime import datetime, timedelta, timezone

from me_core.event_stream import EventHistory, EventStream
from me_core.types import AgentEvent, EventKind


class EventStreamTestCase(unittest.TestCase):
    """EventStream 的追加、过滤与查询行为测试。"""

    def _event(
        self,
        kind: str,
        source: str,
        seconds_offset: int = 0,
    ) -> AgentEvent:
        ts = datetime.now(timezone.utc) + timedelta(seconds=seconds_offset)
        return AgentEvent(
            timestamp=ts,
            event_type=kind,
            payload={"kind": kind},
            source=source,
        )

    def test_append_and_iter_events(self) -> None:
        stream = EventStream(max_events=10)

        e1 = self._event(EventKind.PERCEPTION.value, "human")
        e2 = self._event(EventKind.TOOL_CALL.value, "agent_internal")
        e3 = self._event(EventKind.TOOL_RESULT.value, "tool")

        stream.append_event(e1)
        stream.append_event(e2)
        stream.append_event(e3)

        all_events = list(stream.iter_events())
        self.assertEqual(len(all_events), 3)

        human_events = list(stream.iter_events(source="human"))
        self.assertEqual(human_events, [e1])

        tool_result_events = list(
            stream.iter_events(kind=EventKind.TOOL_RESULT.value)
        )
        self.assertEqual(tool_result_events, [e3])

    def test_last_event_filtering(self) -> None:
        stream = EventStream(max_events=5)

        e1 = self._event("perception", "human", seconds_offset=-2)
        e2 = self._event("perception", "env", seconds_offset=-1)
        stream.append_event(e1)
        stream.append_event(e2)

        last_any = stream.last_event()
        self.assertEqual(last_any, e2)

        last_human = stream.last_event(source="human")
        self.assertEqual(last_human, e1)


class EventHistoryTestCase(unittest.TestCase):
    """EventHistory 的存储与摘要行为测试。"""

    def test_history_add_and_summarize(self) -> None:
        history = EventHistory(max_events=3)

        base_ts = datetime.now(timezone.utc)
        events = [
            AgentEvent(
                timestamp=base_ts,
                event_type="perception",
                payload={"kind": "perception"},
                source="human",
            ),
            AgentEvent(
                timestamp=base_ts + timedelta(seconds=1),
                event_type="tool_result",
                payload={"kind": "tool_result"},
                source="tool",
            ),
            AgentEvent(
                timestamp=base_ts + timedelta(seconds=2),
                event_type="dialogue",
                payload={"kind": "dialogue"},
                source="agent_internal",
            ),
        ]

        for e in events:
            history.add(e)

        # 超过 max_events 时，应自动丢弃最旧的事件
        history.add(
            AgentEvent(
                timestamp=base_ts + timedelta(seconds=3),
                event_type="perception",
                payload={"kind": "perception"},
                source="env",
            )
        )

        recent = history.recent()
        self.assertLessEqual(len(recent), 3)

        summary = history.summarize()
        self.assertIn("total", summary)
        self.assertIn("by_kind", summary)
        self.assertIn("by_source", summary)
        self.assertIsInstance(summary["by_kind"], dict)
        self.assertIsInstance(summary["by_source"], dict)


if __name__ == "__main__":
    unittest.main()

