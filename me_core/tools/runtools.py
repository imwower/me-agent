from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from me_core.tools.base import ToolSpec
from me_core.workspace import Workspace


@dataclass(slots=True)
class RunCommandTool:
    workspace: Workspace
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
        rc, out, err = repo.run_command(cmd, timeout=timeout)
        return {"returncode": rc, "stdout": out, "stderr": err}


@dataclass(slots=True)
class RunTestsTool:
    workspace: Workspace
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
        rc, out, err = repo.run_command(cmd)
        success = rc == 0
        summary = out.splitlines()[:3]
        return {"success": success, "summary": "\n".join(summary), "stdout": out + err}


@dataclass(slots=True)
class RunTrainingScriptTool:
    workspace: Workspace
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
        rc, out, err = repo.run_command(cmd)
        success = rc == 0
        return {"success": success, "stdout": out + err}


__all__ = ["RunCommandTool", "RunTestsTool", "RunTrainingScriptTool"]
