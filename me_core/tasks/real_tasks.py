from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from .types import Scenario, TaskStep


def load_real_task_records(path: str) -> List[Dict[str, Any]]:
    fp = Path(path)
    if not fp.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except Exception:
            continue
    return records


def build_real_task_scenarios(path: str = "data/real_tasks/tasks.jsonl") -> List[Scenario]:
    records = load_real_task_records(path)
    scenarios: List[Scenario] = []
    for rec in records:
        task_id = str(rec.get("id") or f"real_{len(scenarios)}")
        description = rec.get("description") or "真实任务"
        expected = rec.get("expected_keywords") or []
        structured = rec.get("structured_input")
        image_path = rec.get("image_path")
        audio_path = rec.get("audio_path")
        video_path = rec.get("video_path")
        user_prompt = rec.get("question") or rec.get("instruction") or description
        step = TaskStep(
            user_input=user_prompt,
            image_path=image_path,
            audio_path=audio_path,
            video_path=video_path,
            structured_input=structured,
            expected_keywords=expected,
            eval_config={"mode": "contains_any"},
        )
        scenarios.append(
            Scenario(
                id=f"real_{task_id}",
                name=f"RealTask {task_id}",
                description=description,
                steps=[step],
                eval_config={"case_insensitive": True},
            )
        )
    return scenarios


__all__ = ["build_real_task_scenarios", "load_real_task_records"]
