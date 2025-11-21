from __future__ import annotations

import unittest

from me_core.types import AgentEvent, EventKind
from me_core.world_model import SimpleWorldModel


class WorldModelTimelineTestCase(unittest.TestCase):
    def test_timeline_and_queries(self) -> None:
        world = SimpleWorldModel()

        step1 = world.advance_step()
        event1 = AgentEvent.now(
            event_type=EventKind.PERCEPTION.value,
            payload={"raw": {"text": "hello"}},
            kind=EventKind.PERCEPTION,
        )
        event1.modality = "text"
        world.append_event(event1)

        world.advance_step()
        concept = world.concept_space.add_concept("concept_hello", [1.0, 0.0])
        event2 = AgentEvent.now(
            event_type=EventKind.PERCEPTION.value,
            payload={"raw": {"text": "world"}},
            kind=EventKind.PERCEPTION,
        )
        event2.modality = "text"
        event2.meta["concept_id"] = str(concept.id)
        event2.tags.add("greet")
        world.append_event(event2)

        recent = world.recent_events(max_count=1)
        self.assertEqual(recent[0].step, step1 + 1)
        self.assertEqual(recent[0].event, event2)

        filtered = world.query_events(tag="greet")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].event.id, event2.id)

        top = world.top_concepts(top_k=1)
        self.assertTrue(top)
        self.assertEqual(top[0][0].id, concept.id)
        self.assertEqual(top[0][1].modalities.get("text"), 1)

        top_text = world.concepts_by_modality("text")
        self.assertEqual(top_text[0][0].id, concept.id)


if __name__ == "__main__":
    unittest.main()
