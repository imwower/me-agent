from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from me_core.population.population import AgentPopulation
from me_core.policy.agents import load_agent_spec_from_files
from me_core.population.coevo import CoEvoPlanner
from me_core.tasks.generated.pool import TaskPool
from me_core.memory.log_index import LogIndex


class CoEvoPlannerTestCase(unittest.TestCase):
    def test_state_transition(self) -> None:
        pop = AgentPopulation()
        pop.register(load_agent_spec_from_files("a1", None, None))
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = TaskPool(tmpdir)
            log = Path(tmpdir) / "log.jsonl"
            log.write_text("{}", encoding="utf-8")
            planner = CoEvoPlanner(pop, pool, LogIndex(tmpdir))
            state = planner.propose_next_round(None, {"score": 0.5})
            self.assertEqual(state.generation, 0)
            self.assertTrue(state.snn_train_schedules)


if __name__ == "__main__":
    unittest.main()
