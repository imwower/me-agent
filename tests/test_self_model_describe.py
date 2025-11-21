from __future__ import annotations

import unittest

from me_core.self_model import SimpleSelfModel
from me_core.types import AgentEvent, EventKind
from me_core.world_model import SimpleWorldModel


class SelfModelDescribeTestCase(unittest.TestCase):
    def test_describe_self_contains_modalities_and_concepts(self) -> None:
        world = SimpleWorldModel()
        self_model = SimpleSelfModel()

        step = world.advance_step()
        event = AgentEvent.now(
            event_type=EventKind.PERCEPTION.value,
            payload={"raw": {"text": "测试自述"}},
            kind=EventKind.PERCEPTION,
        )
        event.modality = "text"
        concept = world.concept_space.add_concept("测试概念", [1.0, 0.0, 0.0])
        event.meta["concept_id"] = str(concept.id)
        world.append_event(event)
        self_model.observe_event(event, step=step)
        self_model.register_capability_tag("time_tool")  # type: ignore[attr-defined]

        desc = self_model.describe_self(world_model=world)
        self.assertIn("text", desc)
        self.assertIn("time_tool", desc)
        self.assertIn("测试概念", desc)


if __name__ == "__main__":
    unittest.main()
