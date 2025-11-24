from __future__ import annotations

import unittest

from me_core.memory import EpisodicMemory, SemanticMemory, JsonlMemoryStorage
from me_core.self_model import SimpleSelfModel
from me_core.self_model.self_report import generate_long_term_report
from me_core.world_model import SimpleWorldModel


class SelfLongTermReportTestCase(unittest.TestCase):
    def test_generate_long_term_report(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            st = JsonlMemoryStorage(Path(tmpdir) / "eps.jsonl")
            episodic = EpisodicMemory(st)
            semantic = SemanticMemory(st)
            world = SimpleWorldModel()
            self_model = SimpleSelfModel()
            report = generate_long_term_report(self_model.get_state(), world, episodic, semantic)
            self.assertIn("长期自我总结", report)


if __name__ == "__main__":
    unittest.main()
