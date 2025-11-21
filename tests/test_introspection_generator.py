from __future__ import annotations

import unittest

from me_core.introspection import IntrospectionGenerator
from me_core.learning import SimpleLearner
from me_core.self_model import SimpleSelfModel
from me_core.types import AgentEvent, EventKind
from me_core.world_model import SimpleWorldModel


class IntrospectionGeneratorTestCase(unittest.TestCase):
    def test_generate_log(self) -> None:
        world = SimpleWorldModel()
        self_model = SimpleSelfModel()
        learner = SimpleLearner()
        gen = IntrospectionGenerator(world, self_model, learner)

        world.advance_step()
        event = AgentEvent.now(
            event_type=EventKind.PERCEPTION.value,
            payload={"raw": {"text": "测试自省"}},
            kind=EventKind.PERCEPTION,
        )
        event.modality = "text"
        world.append_event(event)
        self_model.observe_event(event, step=world.current_step)
        learner.observe_tool_result("demo_tool", success=False)
        learner.observe_tool_result("demo_tool", success=False)

        log = gen.generate(scenario_id="demo", start_step=1, end_step=world.current_step)
        self.assertIn("本段内", log.summary)
        self.assertTrue(log.mistakes)


if __name__ == "__main__":
    unittest.main()
