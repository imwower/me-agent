from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

from .types import FileEdit, RepoSpec, RepoStatus

logger = logging.getLogger(__name__)


class Repo:
    """对单个本地仓库的受限访问封装。"""

    def __init__(self, spec: RepoSpec) -> None:
        self.spec = spec
        self.path = Path(spec.path).resolve()
        self.allowed = [self.path / p for p in spec.allowed_paths]

    def _assert_allowed(self, relpath: str) -> Path:
        target = (self.path / relpath).resolve()
        for ap in self.allowed:
            try:
                target.relative_to(ap.resolve())
                return target
            except ValueError:
                continue
        raise PermissionError(f"path not allowed: {relpath}")

    def read_file(self, relpath: str, max_bytes: int = 20000) -> str:
        p = self._assert_allowed(relpath)
        data = p.read_bytes()
        if len(data) > max_bytes:
            data = data[:max_bytes]
        return data.decode("utf-8", errors="ignore")

    def write_file(self, relpath: str, content: str) -> None:
        p = self._assert_allowed(relpath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        logger.info("write_file: %s", p)

    def apply_edits(self, edits: Iterable[FileEdit]) -> None:
        for edit in edits:
            p = self._assert_allowed(edit.path)
            text = p.read_text(encoding="utf-8")
            if edit.old_snippet and edit.old_snippet in text:
                text = text.replace(edit.old_snippet, edit.new_snippet, 1)
            else:
                text += "\n" + edit.new_snippet
            p.write_text(text, encoding="utf-8")
            logger.info("apply_edit: %s reason=%s", p, edit.reason)

    def run_command(self, cmd: List[str], timeout: int | None = None) -> Tuple[int, str, str]:
        proc = subprocess.run(
            cmd,
            cwd=self.path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def get_status(self) -> RepoStatus:
        branch = "unknown"
        dirty = False
        last_commit = None
        try:
            ret, out, _ = self.run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            if ret == 0:
                branch = out.strip()
            ret, out, _ = self.run_command(["git", "status", "--porcelain"])
            if ret == 0:
                dirty = bool(out.strip())
            ret, out, _ = self.run_command(["git", "rev-parse", "HEAD"])
            if ret == 0:
                last_commit = out.strip()
        except Exception:
            pass
        return RepoStatus(branch=branch, dirty=dirty, last_commit=last_commit)
