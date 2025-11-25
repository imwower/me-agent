from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

from me_core.tasks.generated.types import GeneratedTask


@dataclass
class TrainSchedule:
    id: str
    repo_id: str
    tasks: List[GeneratedTask]
    config_path: str
    output_dir: str
    max_epochs: int
    priority: int = 0
    max_steps: int | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "config_path": self.config_path,
            "output_dir": self.output_dir,
            "max_epochs": self.max_epochs,
            "priority": self.priority,
            "max_steps": self.max_steps,
            "tasks": [task_to_dict(t) for t in self.tasks],
        }


def export_tasks_for_snn(tasks: List[GeneratedTask], output_dir: str) -> str:
    """
    将生成的任务写出为 self-snn 可消费的占位数据文件，返回路径。
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generated_tasks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(
                json.dumps(
                    {
                        "id": t.id,
                        "kind": t.kind,
                        "payload": t.payload,
                        "expected": t.expected_behavior,
                        "labels": t.labels,
                        "difficulty": t.difficulty,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return str(out_path)


def task_to_dict(task: GeneratedTask) -> dict:
    return {
        "id": task.id,
        "template_id": task.template_id,
        "payload": task.payload,
        "expected_behavior": task.expected_behavior,
        "labels": task.labels,
        "meta": task.meta,
        "difficulty": task.difficulty,
        "kind": task.kind,
    }


def dump_train_schedule(schedule: TrainSchedule, path: str) -> str:
    data = schedule.to_dict()
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
