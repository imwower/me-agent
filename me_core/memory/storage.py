from __future__ import annotations

import json
from pathlib import Path
from typing import List, Protocol, Optional

from .episodic import Episode
from .semantic import ConceptMemory


class MemoryStorage(Protocol):
    def save_episode(self, episode: Episode) -> None:
        ...

    def load_episodes(self, limit: int | None = None) -> List[Episode]:
        ...

    def save_concept_memory(self, cm: ConceptMemory) -> None:
        ...

    def load_concept_memories(self) -> List[ConceptMemory]:
        ...


class JsonlMemoryStorage:
    """使用 JSONL 文件持久化记忆。"""

    def __init__(self, episodes_path: Path, concepts_path: Optional[Path] = None) -> None:
        self.episodes_path = episodes_path
        self.concepts_path = concepts_path or episodes_path.with_suffix(".concepts.jsonl")
        self.episodes_path.parent.mkdir(parents=True, exist_ok=True)
        self.concepts_path.parent.mkdir(parents=True, exist_ok=True)

    def save_episode(self, episode: Episode) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")

    def load_episodes(self, limit: int | None = None) -> List[Episode]:
        if not self.episodes_path.exists():
            return []
        items: List[Episode] = []
        with self.episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                items.append(Episode.from_dict(data))
        if limit is None or limit >= len(items):
            return items
        return items[-limit:]

    def save_concept_memory(self, cm: ConceptMemory) -> None:
        with self.concepts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(cm.to_dict(), ensure_ascii=False) + "\n")

    def load_concept_memories(self) -> List[ConceptMemory]:
        if not self.concepts_path.exists():
            return []
        mems: List[ConceptMemory] = []
        with self.concepts_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                mems.append(ConceptMemory.from_dict(data))
        return mems


__all__ = ["MemoryStorage", "JsonlMemoryStorage"]
