from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from me_core.tools.base import ToolSpec, BaseTool
from me_core.workspace import Workspace


@dataclass(slots=True)
class ReadFileTool:
    workspace: Workspace
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="read_file",
            description="读取指定仓库的文件内容（截断）",
            input_schema={"repo_id": "string", "path": "string"},
            output_schema={"content": "string"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        repo = self.workspace.get_repo(params["repo_id"])
        content = repo.read_file(params["path"])
        return {"content": content}


@dataclass(slots=True)
class WriteFileTool:
    workspace: Workspace
    max_lines: int = 500
    max_files_per_run: int = 10
    _write_count: int = 0
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="write_file",
            description="写入指定仓库文件（全量替换）",
            input_schema={"repo_id": "string", "path": "string", "content": "string"},
            output_schema={"ok": "bool", "message": "string"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        repo = self.workspace.get_repo(params["repo_id"])
        self._write_count += 1
        if self.max_files_per_run and self._write_count > self.max_files_per_run:
            return {"ok": False, "message": "write limit exceeded"}
        content = params.get("content", "")
        if self.max_lines and content.count("\n") > self.max_lines:
            return {"ok": False, "message": "content too long"}
        repo.write_file(params["path"], content)
        return {"ok": True, "message": "written"}


@dataclass(slots=True)
class ApplyPatchTool:
    workspace: Workspace
    max_lines: int = 500
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="apply_patch",
            description="对指定文件应用简单替换",
            input_schema={"repo_id": "string", "path": "string", "old": "string", "new": "string", "reason": "string"},
            output_schema={"ok": "bool"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from me_core.workspace import FileEdit

        if self.max_lines and params.get("new", "").count("\n") > self.max_lines:
            return {"ok": False}
        edit = FileEdit(
            path=params["path"],
            old_snippet=params.get("old", ""),
            new_snippet=params.get("new", ""),
            reason=params.get("reason", "apply_patch"),
        )
        repo = self.workspace.get_repo(params["repo_id"])
        repo.apply_edits([edit])
        return {"ok": True}


__all__ = ["ReadFileTool", "WriteFileTool", "ApplyPatchTool"]
