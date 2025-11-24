"""检查 workspace / agent 配置的健康状况。"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.config import load_agent_config


def _load_workspace(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="配置健康检查")
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--agent-config", type=str, default=None)
    args = parser.parse_args()

    workspace = _load_workspace(Path(args.workspace))
    agent_cfg = load_agent_config(args.agent_config)
    errors: List[str] = []
    warnings: List[str] = []

    for repo in workspace.get("repos", []):
        path = Path(repo.get("path", ""))
        if not path.exists():
            errors.append(f"repo {repo.get('id')} path 不存在: {path}")
        meta = repo.get("meta", {}) or {}
        tags = set(repo.get("tags", []))
        if "brain" in tags:
            for key in ["structure_script", "energy_script", "memory_script"]:
                if key not in meta:
                    warnings.append(f"brain repo {repo.get('id')} 缺少 {key}")
        if "experiment_target" in tags:
            if not meta.get("default_train_cmd"):
                warnings.append(f"experiment repo {repo.get('id')} 缺少 default_train_cmd")
            if not meta.get("default_eval_cmd"):
                warnings.append(f"experiment repo {repo.get('id')} 缺少 default_eval_cmd")

    if agent_cfg.embedding_backend_module:
        try:
            importlib.import_module(agent_cfg.embedding_backend_module)
        except Exception as exc:
            errors.append(f"无法导入 embedding_backend_module: {exc}")

    if errors:
        print("发现错误:")  # noqa: T201
        for e in errors:
            print(f"- {e}")  # noqa: T201
        exit(1)
    if warnings:
        print("警告:")  # noqa: T201
        for w in warnings:
            print(f"- {w}")  # noqa: T201
    print("配置检查完成，无致命错误。")  # noqa: T201


if __name__ == "__main__":
    main()
