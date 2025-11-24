from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from .types import GeneratedTask


class TaskPool:
    def __init__(self, root: str) -> None:
        self.root = Path(root)

    def _iter_files(self) -> List[Path]:
        if not self.root.exists():
            return []
        return list(self.root.glob("**/*.json")) + list(self.root.glob("**/*.jsonl"))

    def list_tasks(self, kind: Optional[str] = None) -> List[GeneratedTask]:
        tasks: List[GeneratedTask] = []
        for fp in self._iter_files():
            if fp.suffix == ".jsonl":
                for line in fp.read_text(encoding="utf-8").splitlines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        gt = self._from_dict(data)
                        if kind is None or gt.kind == kind:
                            tasks.append(gt)
                    except Exception:
                        continue
            else:
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    gt = self._from_dict(data)
                    if kind is None or gt.kind == kind:
                        tasks.append(gt)
                except Exception:
                    continue
        return tasks

    def sample_tasks(
        self,
        kind: Optional[str] = None,
        difficulty_range: Optional[Tuple[int, int]] = None,
        max_count: int = 10,
    ) -> List[GeneratedTask]:
        tasks = self.list_tasks(kind)
        if difficulty_range:
            lo, hi = difficulty_range
            tasks = [t for t in tasks if lo <= t.difficulty <= hi]
        return tasks[:max_count]

    def _from_dict(self, data: dict) -> GeneratedTask:
        return GeneratedTask(
            id=data.get("id", ""),
            template_id=data.get("template_id", ""),
            payload=data.get("payload", {}),
            expected_behavior=data.get("expected_behavior", ""),
            labels=data.get("labels", {}),
            meta=data.get("meta", {}),
            difficulty=int(data.get("difficulty", 1)),
            kind=data.get("kind"),
        )
