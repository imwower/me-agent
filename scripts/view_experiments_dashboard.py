"""CLI 文本仪表盘：汇总实验 / 场景分数与脑指标。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="查看实验仪表盘（文本版）")
    parser.add_argument("--report", type=str, default="outputs/devloop_report.jsonl")
    args = parser.parse_args()

    records = _load_jsonl(Path(args.report))
    if not records:
        print("未找到报告，路径是否正确？")  # noqa: T201
        return

    scenario_scores: Dict[str, List[float]] = defaultdict(list)
    for rec in records:
        sid = rec.get("scenario_id") or rec.get("experiment_id") or "unknown"
        if "score" in rec:
            scenario_scores[sid].append(float(rec.get("score", 0.0)))

    print("=== 场景/实验分数概览 ===")  # noqa: T201
    for sid, scores in scenario_scores.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        latest = scores[-1]
        print(f"- {sid}: 最近 {latest:.3f}，均值 {avg:.3f}，次数 {len(scores)}")  # noqa: T201

    print("\n=== 脑指标/实验备注 ===")  # noqa: T201
    for rec in records[-5:]:
        brain = rec.get("brain") or rec.get("brain_snapshots")
        if brain:
            print(f"- 记录含脑态: {brain}")  # noqa: T201


if __name__ == "__main__":
    main()
