from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Dict

from me_core.self_model import SimpleSelfModel
from me_core.world_model import SimpleWorldModel
from me_core.memory.log_index import LogIndex


class StatusHandler(BaseHTTPRequestHandler):
    world_model: SimpleWorldModel | None = None
    self_model: SimpleSelfModel | None = None
    log_index: LogIndex | None = None

    def _send(self, code: int, data: Dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/status":
            self._send(
                200,
                {
                    "self": self.self_model.describe_self() if self.self_model else "",
                    "world": self.world_model.summarize() if self.world_model else {},
                },
            )
            return
        if self.path.startswith("/experiments/recent"):
            data = []
            if self.log_index:
                data = self.log_index.query(kinds=["experiment"], max_results=10)
            self._send(200, {"items": data})
            return
        self._send(404, {"error": "not found"})


def serve_http(world: SimpleWorldModel, self_model: SimpleSelfModel, log_root: str = "logs", port: int = 8000) -> Thread:
    StatusHandler.world_model = world
    StatusHandler.self_model = self_model
    StatusHandler.log_index = LogIndex(log_root)
    server = HTTPServer(("0.0.0.0", port), StatusHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


__all__ = ["serve_http"]
