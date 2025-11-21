"""读取事件时间线 JSONL 并打印概要。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from me_core.types import AgentEvent


def load_events(path: Path) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            events.append(AgentEvent.from_dict(data))
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="查看事件时间线 JSONL")
    parser.add_argument("path", type=str, help="事件 JSONL 路径")
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        print(f"文件不存在: {path}")  # noqa: T201
        return

    events = load_events(path)
    for ev in events:
        step = ev.meta.get("step") if isinstance(ev.meta, dict) else "-"
        payload = ev.payload or {}
        text = ""
        if isinstance(payload, dict):
            raw = payload.get("raw")
            if isinstance(raw, dict) and isinstance(raw.get("text"), str):
                text = raw["text"][:60]
            elif isinstance(payload.get("text"), str):
                text = payload["text"][:60]
        concept_id = ""
        if isinstance(ev.meta, dict):
            concept_id = str(ev.meta.get("concept_id") or "")
        print(f"[step {step}] {ev.event_type} ({ev.modality}) concept={concept_id} text={text}")  # noqa: T201


if __name__ == "__main__":
    main()
