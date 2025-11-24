from __future__ import annotations

import http.client
import json
import logging
import subprocess
from typing import Any, Dict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class CodeLLMClient:
    """
    一个面向“写/改代码”场景的外部 LLM 客户端占位实现。

    支持三种模式：
    - mock：直接返回配置中的 mock_response，便于离线测试；
    - http：使用 http.client 调用一个简单的 HTTP 接口（POST JSON: {prompt, max_tokens}）；
    - cli：通过 subprocess 调用外部命令（如本地脚本），stdin 输入 prompt，stdout 读取结果。
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.output_format = self.config.get("output_format", "files")
        self.max_retries = int(self.config.get("max_retries", 1))

    def _call_http(self, prompt: str, max_tokens: int) -> str:
        endpoint = self.config.get("endpoint") or ""
        parsed = urlparse(endpoint)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=15)
        path = parsed.path or "/"
        payload = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode("utf-8")
        try:
            conn.request("POST", path, body=payload, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read().decode("utf-8", errors="ignore")
            return data
        except Exception as exc:  # pragma: no cover - 依赖外部 HTTP
            logger.warning("CodeLLMClient http 调用失败: %s", exc)
            return ""
        finally:
            conn.close()

    def _call_cli(self, prompt: str, max_tokens: int) -> str:
        cmd = self.config.get("command") or "cat"
        if isinstance(cmd, str):
            cmd_list = cmd.split()
        else:
            cmd_list = list(cmd)
        try:
            proc = subprocess.Popen(
                cmd_list,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = proc.communicate(prompt.encode("utf-8"), timeout=15)
            return stdout.decode("utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover - 依赖外部命令
            logger.warning("CodeLLMClient cli 调用失败: %s", exc)
            return ""

    def complete(self, prompt: str, max_tokens: int = 2048) -> str:
        mode = self.config.get("mode", "mock")
        if mode == "mock":
            return str(self.config.get("mock_response", ""))
        if mode == "http":
            return self._call_http(prompt, max_tokens)
        if mode == "cli":
            return self._call_cli(prompt, max_tokens)
        # fallback: 回显截断的 prompt，保证流程不中断
        return prompt[-max_tokens:]


def create_client(config: Dict[str, Any] | None = None) -> CodeLLMClient:
    return CodeLLMClient(config or {})
