import unittest

from me_ext.http_api.server import StatusHandler
from me_core.world_model import SimpleWorldModel
from me_core.self_model import SimpleSelfModel


class HttpApiRunTaskTrainStubTest(unittest.TestCase):
    def test_handle_task_run_stub(self) -> None:
        handler = StatusHandler.__new__(StatusHandler)
        handler.agent = None
        handler.world_model = SimpleWorldModel()
        handler.self_model = SimpleSelfModel()
        captured = {}
        handler._send = lambda code, data: captured.update({"code": code, "data": data})  # type: ignore[method-assign]
        handler._handle_task_run({"input": {"text": "hello", "structured": {"cpu": 0.5}}})
        self.assertEqual(captured["code"], 200)
        self.assertIn("reply", captured["data"])

    def test_handle_train_run_noop(self) -> None:
        handler = StatusHandler.__new__(StatusHandler)
        handler.agent = None
        handler.world_model = SimpleWorldModel()
        handler.self_model = SimpleSelfModel()
        captured = {}
        handler._send = lambda code, data: captured.update({"code": code, "data": data})  # type: ignore[method-assign]
        handler._handle_train_run({"mode": "backend"})
        self.assertEqual(captured["data"]["status"], "noop")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
