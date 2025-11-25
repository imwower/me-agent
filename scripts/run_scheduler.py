"""简易长周期调度器：按配置定时运行 job。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from me_core.scheduler.runner import JobRunner, load_jobs, should_run
from me_core.scheduler.types import Job
from me_core.workspace import Workspace
from scripts.run_orchestrator import main as orchestrator_main


def orchestrator_entry(job: Job, ws: Workspace) -> dict[str, Any]:
    # 复用 run_orchestrator，通过子进程样式调用 main 函数比较复杂，直接调用内部函数不可行，这里简单写入配置由 orchestrator 脚本读取
    # 为 demo 简化：仅记录 job 配置，真实执行可调用 subprocess 运行 run_orchestrator.py
    return {"message": f"job {job.id} queued", "config": job.config}


def main() -> None:
    parser = argparse.ArgumentParser(description="长周期调度器")
    parser.add_argument("--jobs", type=str, required=True, help="jobs JSON 路径")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--once", action="store_true", help="仅跑一轮")
    parser.add_argument("--verbose", action="store_true", help="打印调度过程便于观察")
    args = parser.parse_args()

    workspace = Workspace.from_json(args.workspace) if args.workspace else Workspace([])
    jobs = load_jobs(Path(args.jobs))
    runner = JobRunner(workspace, orchestrator_entry)
    last_run: dict[str, float] = {}

    while True:
        now_ts = time.time()
        for job in jobs:
            if should_run(job.schedule, last_run.get(job.id), now_ts):
                if args.verbose:
                    print(f"[scheduler] run job={job.id} schedule={job.schedule} at {now_ts}")
                res = runner.run_job(job)
                if args.verbose:
                    print(f"[scheduler] result job={job.id} -> {res}")
                log_path = Path("logs/jobs")
                log_path.mkdir(parents=True, exist_ok=True)
                out = log_path / f"{job.id}.jsonl"
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                last_run[job.id] = now_ts
            elif args.verbose:
                print(f"[scheduler] skip job={job.id} schedule={job.schedule}")
        if args.once:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
