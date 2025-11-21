from __future__ import annotations

import unittest

from me_core.drives import Intent
from me_core.learning import SimpleLearner


class LearningToolStatsTestCase(unittest.TestCase):
    def test_observe_tool_and_intent(self) -> None:
        learner = SimpleLearner()
        learner.observe_tool_result("demo_tool", True)
        learner.observe_tool_result("demo_tool", False)

        stats = learner.tool_stats["demo_tool"]
        self.assertEqual(stats.call_count, 2)
        self.assertEqual(stats.success_count, 1)

        intent = Intent(kind="reply", priority=1)
        learner.observe_intent_outcome(intent, True)
        intent_stats = learner.intent_stats["reply"]
        self.assertEqual(intent_stats.tried, 1)
        self.assertEqual(intent_stats.succeeded, 1)

        recs = learner.get_tool_recommendations()
        self.assertEqual(recs[0][0], "demo_tool")


if __name__ == "__main__":
    unittest.main()
