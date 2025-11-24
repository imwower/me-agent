from __future__ import annotations

import json
import urllib.request
import unittest

from me_core.memory.log_index import LogIndex
from me_core.research.notebook_builder import NotebookBuilder
from me_core.research.comparison_builder import ComparisonBuilder
from me_core.research.paper_builder import PaperDraftBuilder
from me_core.teachers.manager import TeacherManager
from me_core.teachers.interface import DummyTeacher
from me_core.world_model import SimpleWorldModel
from me_core.self_model import SimpleSelfModel
from me_ext.http_api.server import StatusHandler, HTTPServer, Thread


class HttpApiReportsTestCase(unittest.TestCase):
    def test_reports_endpoints(self) -> None:
        # 重用 serve_http 逻辑，直接构造 handler 的依赖
        StatusHandler.world_model = SimpleWorldModel()
        StatusHandler.self_model = SimpleSelfModel()
        StatusHandler.log_index = LogIndex("logs")
        nb = NotebookBuilder(StatusHandler.log_index, StatusHandler.world_model, StatusHandler.self_model)
        comp = ComparisonBuilder(StatusHandler.log_index)
        StatusHandler.notebook_builder = nb
        StatusHandler.comparison_builder = comp
        StatusHandler.paper_builder = PaperDraftBuilder(nb, comp, TeacherManager([DummyTeacher()]))
        server = HTTPServer(("localhost", 8898), StatusHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        resp = urllib.request.urlopen("http://localhost:8898/report/paper_draft", timeout=2)
        data = json.loads(resp.read().decode("utf-8"))
        self.assertIn("title", data)
        server.shutdown()


if __name__ == "__main__":
    unittest.main()
