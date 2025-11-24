from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .repo import Repo
from .types import RepoSpec


class Workspace:
    """管理多个受限仓库的工作空间。"""

    def __init__(self, specs: List[RepoSpec]) -> None:
        self.specs = {spec.id: spec for spec in specs}
        self.repos: Dict[str, Repo] = {spec.id: Repo(spec) for spec in specs}

    def get_repo(self, repo_id: str) -> Repo:
        if repo_id not in self.repos:
            raise KeyError(f"repo not found: {repo_id}")
        return self.repos[repo_id]

    def list_repos(self) -> List[RepoSpec]:
        return list(self.specs.values())

    def get_repos_by_tag(self, tag: str) -> List[RepoSpec]:
        return [spec for spec in self.specs.values() if tag in spec.tags]

    def get_experiment_targets(self) -> List[RepoSpec]:
        return [spec for spec in self.specs.values() if "experiment_target" in spec.tags]

    def get_brain_repos(self) -> List[RepoSpec]:
        return [spec for spec in self.specs.values() if ("brain" in spec.tags or "snn" in spec.tags)]

    @classmethod
    def from_json(cls, path: str | Path) -> "Workspace":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        specs = []
        for item in data.get("repos", []):
            tags = set(item.get("tags", []) or [])
            specs.append(
                RepoSpec(
                    id=item["id"],
                    name=item.get("name", item["id"]),
                    path=item["path"],
                    allowed_paths=item.get("allowed_paths", ["."]),
                    tags=tags,
                    meta=dict(item.get("meta", {}) or {}),
                )
            )
        return cls(specs)
