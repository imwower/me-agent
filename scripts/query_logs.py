"""简单日志查询 CLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

from me_core.memory.log_index import LogIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="查询 JSONL 日志")
    parser.add_argument("--root", type=str, default="logs")
    parser.add_argument("--kind", action="append", help="日志类型过滤")
    parser.add_argument("--filter", action="append", help="key=value 过滤")
    parser.add_argument("--max-results", type=int, default=50)
    args = parser.parse_args()

    filters = {}
    if args.filter:
        for item in args.filter:
            if "=" in item:
                k, v = item.split("=", 1)
                filters[k] = v

    idx = LogIndex(args.root)
    res = idx.query(kinds=args.kind, filters=filters, max_results=args.max_results)
    for obj in res:
        print(obj)  # noqa: T201


if __name__ == "__main__":
    main()
