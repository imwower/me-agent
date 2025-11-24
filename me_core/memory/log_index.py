from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List


class LogIndex:
    """
    简单的 JSONL 日志索引，用于查询 experiment/devloop/teacher_audit 等记录。
    """

    def __init__(self, root_dir: str) -> None:
        self.root = Path(root_dir)

    def _iter_files(self, kinds: List[str] | None) -> List[Path]:
        if kinds is None:
            return list(self.root.glob("*.jsonl"))
        files: List[Path] = []
        for k in kinds:
            files.extend(self.root.glob(f"*{k}*.jsonl"))
            # 也包含无前缀文件
            files.extend(self.root.glob("*.jsonl"))
        return list({f for f in files})

    def query(
        self,
        kinds: List[str] | None = None,
        since: float | None = None,
        until: float | None = None,
        filters: Dict[str, Any] | None = None,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        results: List[Dict[str, Any]] = []
        for fp in self._iter_files(kinds):
            if not fp.exists():
                continue
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    ts = float(obj.get("ts") or obj.get("time") or time.time())
                    if since and ts < since:
                        continue
                    if until and ts > until:
                        continue
                    passed = True
                    for k, v in filters.items():
                        if obj.get(k) != v:
                            passed = False
                            break
                    if passed:
                        results.append(obj)
                    if len(results) >= max_results:
                        return results
        return results
