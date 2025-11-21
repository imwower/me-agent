"""将 JSONL 形式的事件流打印为简易时间线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from me_core.types import AgentEvent


def load_events(path: Path) -> List[AgentEvent]:
    events: List[AgentEvent] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            events.append(AgentEvent.from_dict(data))
    return events


def summarize_event(event: AgentEvent) -> str:
    payload = event.payload or {}
    text = ""
    if isinstance(payload, dict):
        raw = payload.get("raw")
        if isinstance(raw, dict) and isinstance(raw.get("text"), str):
            text = raw.get("text", "")
        elif isinstance(payload.get("text"), str):
            text = payload.get("text", "")
    if text:
        text = f'"{text[:60]}"'

    if event.event_type == "tool_result":
        success = payload.get("success")
        tool_name = payload.get("tool_name")
        return f"TOOL_RESULT {tool_name} success={success}"
    if event.event_type == "tool_call":
        return f"TOOL_CALL {payload.get('tool_name')}"
    if event.event_type == "dialogue":
        return f"REPLY {text or '(no text)'}"
    return f"{event.event_type.upper()} {text}".strip()


def dump_timeline(events: List[AgentEvent]) -> None:
    by_step: Dict[int, List[AgentEvent]] = {}
    for ev in events:
        step = int(ev.meta.get("step", 0)) if isinstance(ev.meta, dict) else 0
        by_step.setdefault(step, []).append(ev)

    for step in sorted(by_step.keys()):
        print(f"[step {step}]")  # noqa: T201
        for ev in by_step[step]:
            desc = summarize_event(ev)
            print(f"  - {desc}")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description="将 JSONL 事件流转为可读时间线")
    parser.add_argument("path", type=str, help="事件日志 JSONL 路径")
    args = parser.parse_args()
    events = load_events(Path(args.path))
    dump_timeline(events)


if __name__ == "__main__":
    main()
