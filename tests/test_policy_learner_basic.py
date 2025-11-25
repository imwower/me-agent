import unittest

from me_core.learning.policy_learner import PolicyLearner
from me_core.policy.schema import AgentPolicy


class PolicyLearnerBasicTest(unittest.TestCase):
    def test_propose_updates_increase_curiosity(self) -> None:
        learner = PolicyLearner()
        for _ in range(3):
            learner.record_outcome("curiosity.min_concept_count", reward=0.8, success=True)

        policy = AgentPolicy()
        base_value = policy.curiosity.min_concept_count
        updates = learner.propose_updates(policy)

        self.assertIn("curiosity.min_concept_count", updates)
        self.assertGreater(updates["curiosity.min_concept_count"], base_value)

        learner.apply_updates(policy, updates)
        self.assertAlmostEqual(policy.curiosity.min_concept_count, updates["curiosity.min_concept_count"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
