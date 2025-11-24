from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Dict
from pathlib import Path

from me_core.self_model import SimpleSelfModel
from me_core.world_model import SimpleWorldModel
from me_core.memory.log_index import LogIndex
from me_core.research.notebook_builder import NotebookBuilder
from me_core.research.comparison_builder import ComparisonBuilder
from me_core.research.paper_builder import PaperDraftBuilder
from me_core.teachers.manager import TeacherManager
from me_core.teachers.interface import DummyTeacher
import argparse
import time
import sys


class StatusHandler(BaseHTTPRequestHandler):
    world_model: SimpleWorldModel | None = None
    self_model: SimpleSelfModel | None = None
    log_index: LogIndex | None = None
    notebook_builder: NotebookBuilder | None = None
    comparison_builder: ComparisonBuilder | None = None
    paper_builder: PaperDraftBuilder | None = None

    def _send(self, code: int, data: Dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/dashboard"}:
            self._serve_static("/static/index.html")
            return
        if self.path.startswith("/static/"):
            self._serve_static(self.path)
            return
        if self.path.startswith("/plots/"):
            self._serve_plot(self.path.replace("/plots/", ""))
            return
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
        if self.path.startswith("/notebook/recent"):
            if self.notebook_builder:
                nb = self.notebook_builder.build_notebook(max_entries=20)
                brief = [
                    {"id": e.id, "kind": e.kind, "desc": e.description, "metrics": e.metrics, "ts": e.timestamp}
                    for e in nb.entries
                ]
                self._send(200, {"entries": brief})
            else:
                self._send(500, {"error": "notebook_builder not set"})
            return
        if self.path.startswith("/report/comparison"):
            if self.comparison_builder:
                points = self.comparison_builder.build_config_points(top_k=10)
                summary = self.comparison_builder.generate_text_summary(points)
                self._send(200, {"summary": summary, "points": [p.__dict__ for p in points]})
            else:
                self._send(500, {"error": "comparison_builder not set"})
            return
        if self.path.startswith("/report/paper_draft"):
            if self.paper_builder:
                draft = self.paper_builder.build_draft_outline()
                self._send(
                    200,
                    {
                        "title": draft.title,
                        "abstract": draft.abstract,
                        "sections": [{"title": s.title, "content": s.content} for s in draft.sections],
                    },
                )
            else:
                self._send(500, {"error": "paper_builder not set"})
            return
        if self.path.startswith("/plots/list"):
            plots_dir = Path("reports/plots")
            items = [p.name for p in plots_dir.glob("*.png")] if plots_dir.exists() else []
            self._send(200, {"plots": items})
            return
        self._send(404, {"error": "not found"})

    def _serve_static(self, path: str) -> None:
        static_root = Path(__file__).parent / "static"
        rel = path.replace("/static/", "")
        target = static_root / rel
        if not target.exists() or not target.is_file():
            # 回退到 plots 输出目录
            alt = Path("reports/plots") / rel
            target = alt
            if not target.exists() or not target.is_file():
                self._send(404, {"error": "static not found"})
                return
        try:
            data = target.read_bytes()
            content_type = "text/plain"
            if target.suffix == ".html":
                content_type = "text/html"
            elif target.suffix == ".css":
                content_type = "text/css"
            elif target.suffix == ".js":
                content_type = "application/javascript"
            elif target.suffix == ".png":
                content_type = "image/png"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self._send(500, {"error": "failed to read static file"})

    def _serve_plot(self, name: str) -> None:
        target = Path("reports/plots") / name
        if not target.exists() or not target.is_file():
            self._send(404, {"error": "plot not found"})
            return
        try:
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self._send(500, {"error": "failed to read plot"})


def serve_http(world: SimpleWorldModel, self_model: SimpleSelfModel, log_root: str = "logs", port: int = 8000) -> Thread:
    StatusHandler.world_model = world
    StatusHandler.self_model = self_model
    StatusHandler.log_index = LogIndex(log_root)
    StatusHandler.notebook_builder = NotebookBuilder(StatusHandler.log_index, world, self_model)
    comp = ComparisonBuilder(StatusHandler.log_index)
    StatusHandler.comparison_builder = comp
    StatusHandler.paper_builder = PaperDraftBuilder(StatusHandler.notebook_builder, comp, TeacherManager([DummyTeacher()]))
    server = HTTPServer(("0.0.0.0", port), StatusHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 HTTP Dashboard")
    parser.add_argument("--workspace", type=str, default=None, help="当前未使用，仅保留接口兼容性")
    parser.add_argument("--log-root", type=str, default="logs")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    serve_http(SimpleWorldModel(), SimpleSelfModel(), log_root=args.log_root, port=args.port)
    print(f"HTTP server on {args.port}. 按 Ctrl+C 停止。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()


__all__ = ["serve_http", "main"]
