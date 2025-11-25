import unittest

from me_core.types import AgentEvent, EventKind
from me_core.world_model import SimpleWorldModel


class WorldModelTransitionStatsTest(unittest.TestCase):
    def test_transition_stats_and_prediction(self) -> None:
        world = SimpleWorldModel()

        scenario_event = AgentEvent.now("scenario", {"note": "start"})
        scenario_event.meta["scenario_id"] = "sceneA"
        world.append_event(scenario_event)

        call_event = AgentEvent.now(EventKind.TOOL_CALL.value, {"tool_name": "checker"})
        call_event.meta["scenario_id"] = "sceneA"
        world.append_event(call_event)

        result_event = AgentEvent.now(
            EventKind.TOOL_RESULT.value, {"tool_name": "checker", "success": True, "reward": 0.8}
        )
        result_event.meta["scenario_id"] = "sceneA"
        world.append_event(result_event)

        key = ("sceneA", "checker")
        stats = world.transition_stats.get(key)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertGreaterEqual(stats.count, 1)
        self.assertEqual(stats.success_count, 1)

        prob = world.predict_success_prob("sceneA", "checker")
        self.assertGreater(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
