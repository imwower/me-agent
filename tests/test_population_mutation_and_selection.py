from __future__ import annotations

import unittest

from me_core.config import AgentConfig
from me_core.policy import AgentPolicy
from me_core.policy.agents import AgentSpec
from me_core.population.evolution import crossover_policy, mutate_config, mutate_policy, select_top


class PopulationEvolutionTestCase(unittest.TestCase):
    def test_select_top(self) -> None:
        specs = [AgentSpec(id=f"s{i}", config=AgentConfig(), policy=AgentPolicy()) for i in range(3)]
        scores = [0.1, 0.5, 0.3]
        top = select_top(specs, scores, 2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0].id, "s1")

    def test_mutate_policy_and_config(self) -> None:
        policy = AgentPolicy()
        new_policy = mutate_policy(policy)
        self.assertIsInstance(new_policy, AgentPolicy)

        cfg = AgentConfig()
        new_cfg = mutate_config(cfg)
        self.assertIsInstance(new_cfg, AgentConfig)

    def test_crossover_policy(self) -> None:
        p1 = AgentPolicy()
        p2 = AgentPolicy()
        child = crossover_policy(p1, p2)
        self.assertIsInstance(child, AgentPolicy)


if __name__ == "__main__":
    unittest.main()
