"""联合进化 CLI 仪表盘。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 CoEvo 进化报告")
    parser.add_argument("--log", type=str, default="logs/coevo.jsonl")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print("未找到 coevo 日志")  # noqa: T201
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        gen = obj.get("generation")
        scores = obj.get("results", {})
        print(f"第 {gen} 代: scores={scores}")  # noqa: T201


if __name__ == "__main__":
    main()
