"""查看多 Agent 协作消息时间线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="查看协作消息日志")
    parser.add_argument("--log", type=str, required=True, help="包含 messages 字段的 JSONL")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print("未找到日志文件")  # noqa: T201
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        for msg in obj.get("messages", []):
            if isinstance(msg, dict):
                print(f"[{msg.get('from_role')} -> {msg.get('to_role')}] {msg.get('kind')}: {msg.get('content')}")  # noqa: T201


if __name__ == "__main__":
    main()
