from __future__ import annotations

import json
from pathlib import Path
from typing import List

from me_core.codetasks.types import CodeTask


def load_codefix_tasks(root: str) -> List[CodeTask]:
    root_path = Path(root)
    tasks: List[CodeTask] = []
    for spec_path in root_path.glob("*/spec.json"):
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        repo_id = spec.get("repo_id", "codefix")
        files_to_edit = spec.get("files", [])
        description = spec.get("description", "修复 bug")
        test_cmd = spec.get("test_cmd", ["python", "-m", "pytest"])
        tasks.append(
            CodeTask(
                id=spec.get("id", spec_path.parent.name),
                repo_id=repo_id,
                description=description,
                files_to_read=files_to_edit,
                files_to_edit=files_to_edit,
                test_command=test_cmd,
            )
        )
    return tasks
