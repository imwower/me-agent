from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

from me_core.tools.base import ToolSpec
from me_core.workspace import Workspace

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunCommandTool:
    workspace: Workspace
    allowed_prefixes: List[List[str]] = field(default_factory=list)
    blocked_commands: List[str] = field(default_factory=lambda: ["rm", "shutdown", "reboot"])
    default_timeout: int = 300
    max_output_kb: int = 64
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="run_command",
            description="在指定仓库执行命令",
            input_schema={"repo_id": "string", "cmd": "list[str]", "timeout": "int?"},
            output_schema={"returncode": "int", "stdout": "string", "stderr": "string"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        repo = self.workspace.get_repo(params["repo_id"])
        cmd = params.get("cmd") or []
        timeout = params.get("timeout")
        if not cmd:
            return {"returncode": -1, "stdout": "", "stderr": "empty command"}
        if self.blocked_commands and cmd[0] in self.blocked_commands:
            return {"returncode": -1, "stdout": "", "stderr": "blocked command"}
        if self.allowed_prefixes:
            allowed = any(cmd[: len(pref)] == pref for pref in self.allowed_prefixes)
            if not allowed:
                return {"returncode": -1, "stdout": "", "stderr": "command not in whitelist"}
        timeout = timeout or self.default_timeout
        rc, out, err = repo.run_command(cmd, timeout=timeout)
        out = self._truncate(out)
        err = self._truncate(err)
        return {"returncode": rc, "stdout": out, "stderr": err}

    def _truncate(self, text: str) -> str:
        limit = self.max_output_kb * 1024
        if len(text.encode("utf-8")) <= limit:
            return text
        return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore") + "\n...输出已截断..."


@dataclass(slots=True)
class RunTestsTool:
    workspace: Workspace
    default_timeout: int = 600
    max_output_kb: int = 64
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="run_tests",
            description="运行单测命令",
            input_schema={"repo_id": "string", "command": "list[str]?"},
            output_schema={"success": "bool", "summary": "string", "stdout": "string"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        repo = self.workspace.get_repo(params["repo_id"])
        cmd = params.get("command") or ["python", "-m", "unittest"]
        rc, out, err = repo.run_command(cmd, timeout=self.default_timeout)
        success = rc == 0
        summary = (out + err).splitlines()[:3]
        merged = (out + err)
        if len(merged.encode("utf-8")) > self.max_output_kb * 1024:
            merged = merged.encode("utf-8")[: self.max_output_kb * 1024].decode("utf-8", errors="ignore") + "\n...输出已截断..."
        return {"success": success, "summary": "\n".join(summary), "stdout": merged}


@dataclass(slots=True)
class RunTrainingScriptTool:
    workspace: Workspace
    default_timeout: int = 1200
    max_output_kb: int = 64
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="run_training",
            description="运行训练脚本",
            input_schema={"repo_id": "string", "command": "list[str]"},
            output_schema={"success": "bool", "stdout": "string"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        repo = self.workspace.get_repo(params["repo_id"])
        cmd = params.get("command") or []
        rc, out, err = repo.run_command(cmd, timeout=self.default_timeout)
        success = rc == 0
        merged = (out + err)
        if len(merged.encode("utf-8")) > self.max_output_kb * 1024:
            merged = merged.encode("utf-8")[: self.max_output_kb * 1024].decode("utf-8", errors="ignore") + "\n...输出已截断..."
        return {"success": success, "stdout": merged}


__all__ = ["RunCommandTool", "RunTestsTool", "RunTrainingScriptTool"]
