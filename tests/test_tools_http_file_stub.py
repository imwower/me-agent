from __future__ import annotations

import http.server
import socketserver
import threading
import tempfile
import unittest
from pathlib import Path

from me_core.tools import FileReadTool, HttpGetTool


class _QuietHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # type: ignore[override]
        body = b"hello from stub server"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class ToolsHttpFileStubTestCase(unittest.TestCase):
    def test_http_get_tool(self) -> None:
        with socketserver.TCPServer(("localhost", 0), _QuietHandler) as httpd:
            port = httpd.server_address[1]
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            tool = HttpGetTool(timeout=2.0)
            result = tool.run({"url": f"http://localhost:{port}"})
            httpd.shutdown()
        self.assertEqual(result["status_code"], 200)
        self.assertIn("body", result)
        self.assertIn("hello", result["body"])

    def test_file_read_tool(self) -> None:
        tool = FileReadTool(max_length=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "demo.txt"
            path.write_text("0123456789ABC", encoding="utf-8")
            result = tool.run({"path": str(path)})
            self.assertEqual(result["content"], "0123456789")


if __name__ == "__main__":
    unittest.main()
