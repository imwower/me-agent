from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from me_core.teachers.types import ConfigPatch
from me_core.workspace import Workspace


def _set_by_path(obj: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = obj
    for name in parts[:-1]:
        if name not in target or not isinstance(target[name], dict):
            target[name] = {}
        target = target[name]  # type: ignore[assignment]
    target[parts[-1]] = value


def apply_config_patches(workspace: Workspace, patches: List[ConfigPatch]) -> None:
    """
    仅支持 JSON 配置文件的补丁应用。
    对每个 ConfigPatch:
    - 读取文件（UTF-8）并解析 JSON
    - 根据 path 进行点号分割写入
    - 写回文件
    """

    for patch in patches:
        repo = workspace.get_repo(patch.repo_id)
        cfg_path = Path(repo.path) / patch.config_path
        if not cfg_path.exists():
            continue
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            # 不支持非 JSON 或解析失败时跳过
            continue
        if isinstance(data, dict):
            _set_by_path(data, patch.path, patch.value)
            cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["apply_config_patches"]
