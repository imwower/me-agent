from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from me_core.memory.log_index import LogIndex
from me_core.research.notebook_builder import NotebookBuilder
from me_core.world_model import SimpleWorldModel
from me_core.self_model import SimpleSelfModel


class NotebookBuilderTestCase(unittest.TestCase):
    def test_build_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "exp.jsonl"
            log.write_text(json.dumps({"kind": "benchmark", "scenario_id": "s1", "score": 0.8, "ts": 1}), encoding="utf-8")
            idx = LogIndex(tmpdir)
            nb = NotebookBuilder(idx, SimpleWorldModel(), SimpleSelfModel()).build_notebook(max_entries=1)
            self.assertEqual(len(nb.entries), 1)


if __name__ == "__main__":
    unittest.main()
