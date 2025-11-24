from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CodeTask:
    id: str
    repo_id: str
    title: str
    description: str
    files_to_read: List[str] = field(default_factory=list)
    files_to_edit: List[str] = field(default_factory=list)
    test_command: List[str] = field(default_factory=lambda: ["python", "-m", "unittest"])
    constraints: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
