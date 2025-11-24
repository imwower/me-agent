from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from .pool import TaskPool
from .types import GeneratedTask
from me_core.memory.log_index import LogIndex


@dataclass
class CurriculumPolicy:
    mode: Literal["easy2hard", "focus_gaps", "random"] = "easy2hard"
    max_per_round: int = 10


class CurriculumScheduler:
    def __init__(self, pool: TaskPool, log_index: LogIndex) -> None:
        self.pool = pool
        self.log_index = log_index

    def select_next_tasks(self, policy: CurriculumPolicy, recent_results: List[Dict[str, Any]]) -> List[GeneratedTask]:
        if policy.mode == "random":
            return self.pool.sample_tasks(max_count=policy.max_per_round)
        if policy.mode == "focus_gaps":
            bad_kinds = []
            for r in recent_results:
                if float(r.get("score", 1.0)) < 0.6:
                    bad_kinds.append(r.get("kind"))
            kind = bad_kinds[0] if bad_kinds else None
            return self.pool.sample_tasks(kind=kind, max_count=policy.max_per_round)
        # 默认 easy2hard
        tasks = self.pool.list_tasks()
        tasks.sort(key=lambda t: t.difficulty)
        return tasks[: policy.max_per_round]

