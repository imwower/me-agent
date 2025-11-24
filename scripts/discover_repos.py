"""扫描本地目录生成 workspace 配置草稿。"""

from __future__ import annotations

import argparse
from pathlib import Path

from me_core.workspace import scan_local_repo_for_tools
from me_core.workspace.discovery import save_profiles_to_json


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描本地仓库，生成 workspace 草稿")
    parser.add_argument("--root", type=str, required=True, help="待扫描的根目录")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--output", type=str, default="configs/workspace.generated.json")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    profiles = []
    for p in root.glob("*"):
        if not p.is_dir():
            continue
        if p.samefile(root):
            continue
        if p.relative_to(root).as_posix().count("/") >= args.max_depth:
            continue
        try:
            profile = scan_local_repo_for_tools(str(p))
            profiles.append(profile)
        except Exception:
            continue
    save_profiles_to_json(profiles, Path(args.output))
    print(f"已生成 {len(profiles)} 条 repo 配置到 {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
