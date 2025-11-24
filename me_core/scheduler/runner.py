from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from me_core.scheduler.types import Job
from me_core.workspace import Workspace


class JobRunner:
    """
    简易 JobRunner，根据 Job.kind 调用 orchestrator/DevLoop/实验等入口。
    """

    def __init__(self, workspace: Workspace, orchestrator_entry: Callable[[Job, Workspace], dict[str, Any]]) -> None:
        self.workspace = workspace
        self.orchestrator_entry = orchestrator_entry

    def run_job(self, job: Job) -> dict[str, Any]:
        if not job.enabled:
            return {"job_id": job.id, "skipped": True, "reason": "disabled"}
        try:
            res = self.orchestrator_entry(job, self.workspace)
            return {"job_id": job.id, "result": res, "ts": time.time()}
        except Exception as exc:
            return {"job_id": job.id, "error": str(exc), "ts": time.time()}


def should_run(schedule: str, last_run: float | None, now_ts: float) -> bool:
    if schedule == "hourly":
        return last_run is None or (now_ts - last_run) >= 3600
    if schedule == "interval":
        return True
    # default daily
    return last_run is None or (now_ts - last_run) >= 86400


def load_jobs(path: Path) -> list[Job]:
    data = json.loads(path.read_text(encoding="utf-8"))
    jobs: list[Job] = []
    for item in data.get("jobs", []):
        jobs.append(
            Job(
                id=item["id"],
                name=item.get("name", item["id"]),
                kind=item.get("kind", "devloop"),
                config=item.get("config", {}),
                schedule=item.get("schedule", "daily"),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return jobs


__all__ = ["JobRunner", "load_jobs", "should_run"]
