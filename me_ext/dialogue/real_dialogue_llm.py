from __future__ import annotations

import http.client
import json
import logging
import subprocess
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class RealDialogueLLM:
    """
    面向对话输出的 LLM 适配器。
    - 支持 http/cli/stub 模式；
    - prompt 已经在上层构造，这里只负责调用并返回文本。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def _call_http(self, prompt: str) -> str:
        endpoint = self.config.get("endpoint") or ""
        parsed = urlparse(endpoint)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=10)
        path = parsed.path or "/"
        try:
            payload = json.dumps({"prompt": prompt}).encode("utf-8")
            conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            return resp.read().decode("utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - 依赖外部网络
            logger.warning("RealDialogueLLM http 调用失败: %s", exc)
            return ""
        finally:
            conn.close()

    def _call_cli(self, prompt: str) -> str:
        cmd = self.config.get("command") or "cat"
        try:
            proc = subprocess.Popen(
                cmd.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = proc.communicate(prompt.encode("utf-8"), timeout=10)
            return stdout.decode("utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - 依赖外部命令
            logger.warning("RealDialogueLLM cli 调用失败: %s", exc)
            return ""

    def generate_reply(self, prompt: str, meta: Optional[Dict[str, Any]] = None) -> str:
        mode = self.config.get("mode") or "cli"
        if mode == "stub":
            return str(self.config.get("stub_response") or "stub reply")
        if mode == "http":
            return self._call_http(prompt)
        return self._call_cli(prompt)
