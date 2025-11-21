"""查看 Episodic/Semantic Memory 的 JSONL 内容。"""

from __future__ import annotations

import argparse
from pathlib import Path

from me_core.memory import JsonlMemoryStorage


def main() -> None:
    parser = argparse.ArgumentParser(description="读取记忆文件并打印概要")
    parser.add_argument("--episodes", type=str, required=True, help="Episode JSONL 路径")
    parser.add_argument("--concepts", type=str, default=None, help="Concept memory JSONL 路径，可选")
    args = parser.parse_args()

    storage = JsonlMemoryStorage(Path(args.episodes), Path(args.concepts) if args.concepts else None)
    episodes = storage.load_episodes(limit=None)
    concepts = storage.load_concept_memories()

    print("=== Episodes ===")  # noqa: T201
    for ep in episodes[-20:]:
        print(f"- id={ep.id} step={ep.start_step}->{ep.end_step} tags={sorted(ep.tags)} summary={ep.summary}")  # noqa: T201

    print("\n=== Concept Memories ===")  # noqa: T201
    for cm in concepts[-20:]:
        print(f"- {cm.name} ({cm.concept_id}) desc={cm.description[:80]} examples={len(cm.examples)}")  # noqa: T201


if __name__ == "__main__":
    main()
