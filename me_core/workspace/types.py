from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set


@dataclass
class RepoSpec:
    id: str
    name: str
    path: str
    allowed_paths: List[str]
    tags: Set[str] = field(default_factory=set)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileEdit:
    path: str
    old_snippet: str
    new_snippet: str
    reason: str


@dataclass
class RepoStatus:
    branch: str
    dirty: bool
    last_commit: Optional[str]
    last_test_result: Optional[str] = None


@dataclass
class RepoProfile:
    id: str
    name: str
    path: str
    tags: Set[str] = field(default_factory=set)
    detected_tools: List[str] = field(default_factory=list)
    detected_scripts: List[str] = field(default_factory=list)
    notes: str = ""
