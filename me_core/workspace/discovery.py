from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from .types import RepoProfile


def scan_local_repo_for_tools(path: str) -> RepoProfile:
    """
    简单扫描本地仓库，基于 scripts 下的文件名/内容做工具能力推断。
    """

    repo_path = Path(path).resolve()
    profile = RepoProfile(id=repo_path.name, name=repo_path.name, path=str(repo_path))
    scripts_dir = repo_path / "scripts"
    detected_tools: List[str] = []
    detected_scripts: List[str] = []
    if scripts_dir.exists() and scripts_dir.is_dir():
        for p in scripts_dir.glob("**/*"):
            if not p.is_file():
                continue
            if p.suffix not in {".py", ".sh"}:
                continue
            rel = p.relative_to(repo_path).as_posix()
            detected_scripts.append(rel)
            name = p.name.lower()
            text = ""
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            if re.search(r"train", name):
                detected_tools.append("run_train")
            if re.search(r"eval|test", name):
                detected_tools.append("run_eval")
            if "dump" in name or "export" in name:
                detected_tools.append("dump_tool")
            if "brain" in rel or "snn" in rel:
                profile.tags.add("brain")
            if "agent" in rel:
                profile.tags.add("agent")
            if "snn" in text:
                profile.tags.add("snn")
    profile.detected_scripts = sorted(set(detected_scripts))
    profile.detected_tools = sorted(set(detected_tools))
    if not profile.tags:
        profile.tags.add("misc")
    return profile


def generate_workspace_entry_from_profile(profile: RepoProfile) -> Dict[str, Any]:
    """
    将 RepoProfile 转换为 workspace.json 中的一条 repo 配置。
    """

    meta: Dict[str, Any] = {}
    # 简单规则生成默认命令
    for script in profile.detected_scripts:
        if "train" in script and "default_train_cmd" not in meta:
            meta["default_train_cmd"] = ["python", script]
        if "eval" in script and "default_eval_cmd" not in meta:
            meta["default_eval_cmd"] = ["python", script]
        if "dump_brain" in script or "dump_brain_graph" in script:
            meta["structure_script"] = ["python", script]
        if "eval_router_energy" in script:
            meta["energy_script"] = ["python", script, "--json"]
        if "eval_memory" in script:
            meta["memory_script"] = ["python", script, "--json"]
        if "run_brain_infer" in script:
            meta["brain_infer_script"] = ["python", script]
    return {
        "id": profile.id,
        "name": profile.name,
        "path": profile.path,
        "allowed_paths": ["."],
        "tags": list(profile.tags),
        "meta": meta,
    }


def save_profiles_to_json(profiles: List[RepoProfile], output: Path) -> None:
    data = [generate_workspace_entry_from_profile(p) for p in profiles]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"repos": data}, ensure_ascii=False, indent=2), encoding="utf-8")
