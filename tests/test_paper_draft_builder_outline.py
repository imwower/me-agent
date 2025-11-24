from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from me_core.memory.log_index import LogIndex
from me_core.research.notebook_builder import NotebookBuilder
from me_core.research.comparison_builder import ComparisonBuilder
from me_core.research.paper_builder import PaperDraftBuilder
from me_core.teachers.manager import TeacherManager
from me_core.teachers.interface import DummyTeacher
from me_core.world_model import SimpleWorldModel
from me_core.self_model import SimpleSelfModel


class PaperDraftBuilderTestCase(unittest.TestCase):
    def test_build_outline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "exp.jsonl"
            path.write_text(json.dumps({"kind": "experiment", "score": 0.8}), encoding="utf-8")
            idx = LogIndex(tmpdir)
            nb = NotebookBuilder(idx, SimpleWorldModel(), SimpleSelfModel())
            comp = ComparisonBuilder(idx)
            builder = PaperDraftBuilder(nb, comp, TeacherManager([DummyTeacher()]))
            draft = builder.build_draft_outline()
            self.assertTrue(draft.title)
            self.assertTrue(draft.sections)


if __name__ == "__main__":
    unittest.main()
