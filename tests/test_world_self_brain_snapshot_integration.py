from __future__ import annotations

import unittest

from me_core.brain import BrainSnapshot
from me_core.self_model import SimpleSelfModel
from me_core.world_model import SimpleWorldModel


class BrainSnapshotIntegrationTestCase(unittest.TestCase):
    def test_world_and_self_update_from_brain_snapshot(self) -> None:
        snapshot = BrainSnapshot(
            repo_id="brain",
            region_activity={"mcc": 0.1},
            global_metrics={"branching_kappa": 1.05},
            memory_summary={"n_keys": 2},
            decision_hint={"mode": "explore", "confidence": 0.6},
        )
        world = SimpleWorldModel()
        self_model = SimpleSelfModel()

        world.update_brain_snapshot(snapshot)
        self_model.observe_brain_snapshot(snapshot)

        self.assertEqual(world.last_brain_snapshot, snapshot)
        self.assertEqual(self_model.get_state().last_brain_mode, "explore")
        self.assertGreater(self_model.get_state().last_brain_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
