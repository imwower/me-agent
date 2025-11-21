from __future__ import annotations

import unittest

from me_core.drives import SimpleDriveSystem
from me_core.self_model import SimpleSelfModel
from me_core.types import AgentEvent, EventKind
from me_core.world_model import ConceptStats, SimpleWorldModel


class DrivesIntentPriorityTestCase(unittest.TestCase):
    def test_time_request_triggers_tool_call(self) -> None:
        drive = SimpleDriveSystem()
        world = SimpleWorldModel()
        self_model = SimpleSelfModel()

        event = AgentEvent.now(
            event_type=EventKind.PERCEPTION.value,
            payload={"raw": {"text": "现在几点"}},
            kind=EventKind.PERCEPTION,
            source="human",
        )
        intent = drive.decide_intent(self_model, world, [event])
        self.assertEqual(intent.kind, "call_tool")
        self.assertEqual(intent.target_tool, "time")
        self.assertGreaterEqual(intent.priority, 1)

    def test_curiosity_for_single_modality_concept(self) -> None:
        drive = SimpleDriveSystem(curiosity_min_count=1)
        world = SimpleWorldModel()
        self_model = SimpleSelfModel()

        concept = world.concept_space.add_concept("单模态概念", [1.0, 0.0])
        world.concept_stats[concept.id] = ConceptStats(
            count=3,
            modalities={"text": 3},
            last_seen_step=0,
        )

        intent = drive.decide_intent(self_model, world, [])
        self.assertEqual(intent.kind, "curiosity")
        self.assertEqual(intent.target_concept_id, concept.id)


if __name__ == "__main__":
    unittest.main()
