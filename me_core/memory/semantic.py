from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from me_core.alignment.concepts import ConceptId

if False:  # pragma: no cover - 类型提示占位
    from .storage import MemoryStorage


@dataclass
class ConceptMemory:
    concept_id: ConceptId
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "concept_id": str(self.concept_id),
            "name": self.name,
            "description": self.description,
            "examples": list(self.examples),
            "tags": sorted(self.tags),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConceptMemory":
        return cls(
            concept_id=ConceptId(str(data.get("concept_id") or "")),
            name=str(data.get("name") or ""),
            description=str(data.get("description") or ""),
            examples=list(data.get("examples") or []),
            tags=set(data.get("tags") or []),
            updated_at=float(data.get("updated_at") or time.time()),
        )


class SemanticMemory:
    """概念级的长期记忆。"""

    def __init__(self, storage: "MemoryStorage") -> None:
        self.storage = storage
        self._concepts: Dict[str, ConceptMemory] = {}
        try:
            for cm in self.storage.load_concept_memories():
                self._concepts[str(cm.concept_id)] = cm
        except Exception:
            self._concepts = {}

    def upsert_concept_memory(
        self,
        concept_id: ConceptId,
        name: str,
        description: str,
        example: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> None:
        key = str(concept_id)
        cm = self._concepts.get(key)
        if cm is None:
            cm = ConceptMemory(concept_id=concept_id, name=name, description=description)
        else:
            cm.name = name or cm.name
            cm.description = description or cm.description
        if example:
            cm.examples.append(example)
        if tags:
            cm.tags.update(tags)
        cm.updated_at = time.time()
        self._concepts[key] = cm
        try:
            self.storage.save_concept_memory(cm)
        except Exception:
            pass

    def get(self, concept_id: ConceptId) -> Optional[ConceptMemory]:
        return self._concepts.get(str(concept_id))

    def search_by_name(self, keyword: str, max_count: int = 20) -> List[ConceptMemory]:
        keyword_lower = keyword.lower()
        matched = [
            cm
            for cm in self._concepts.values()
            if keyword_lower in cm.name.lower()
        ]
        matched.sort(key=lambda x: x.updated_at, reverse=True)
        return matched[:max_count]

    def all_memories(self) -> List[ConceptMemory]:
        return list(self._concepts.values())
