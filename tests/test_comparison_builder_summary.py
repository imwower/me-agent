from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from me_core.memory.log_index import LogIndex
from me_core.research.comparison_builder import ComparisonBuilder


class ComparisonBuilderTestCase(unittest.TestCase):
    def test_generate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "exp.jsonl"
            path.write_text(json.dumps({"kind": "experiment", "id": "p1", "score": 0.9, "energy": 0.1}), encoding="utf-8")
            builder = ComparisonBuilder(LogIndex(tmpdir))
            points = builder.build_config_points(top_k=5)
            summary = builder.generate_text_summary(points)
            self.assertTrue(summary)


if __name__ == "__main__":
    unittest.main()
