from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from .types import Scenario, TaskStep


def load_multimodal_benchmark(path: str) -> List[Scenario]:
    data = []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    scenarios: List[Scenario] = []
    for item in data:
        keywords = item.get("keywords") or item.get("answers") or []
        scenarios.append(
            Scenario(
                id=str(item.get("id")),
                name=item.get("question", "multimodal"),
                description=item.get("question", ""),
                steps=[
                    TaskStep(
                        user_input=item.get("question", ""),
                        image_path=item.get("image_path"),
                        expected_keywords=keywords,
                        eval_config={"mode": "contains_any"},
                    )
                ],
                eval_config={"case_insensitive": True},
                requires_brain_infer=False,
            )
        )
    return scenarios
