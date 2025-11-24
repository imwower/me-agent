from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

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
    graph_ref: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "concept_id": str(self.concept_id),
            "name": self.name,
            "description": self.description,
            "examples": list(self.examples),
            "tags": sorted(self.tags),
            "updated_at": self.updated_at,
            "graph_ref": self.graph_ref,
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
            graph_ref=data.get("graph_ref"),
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
        graph_ref: Optional[Dict[str, Any]] = None,
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
        if graph_ref:
            cm.graph_ref = graph_ref
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

    # 脑图谱映射：将 BrainGraph 转成多个 ConceptMemory
    def upsert_brain_graph_memory(self, brain_graph: "BrainGraph") -> None:
        try:
            from me_core.alignment.concepts import ConceptNode
        except Exception:
            ConceptNode = None  # type: ignore

        brain_concept_id = ConceptId(f"brain:{brain_graph.repo_id}")
        desc = brain_graph.summary()
        tags = {"brain", brain_graph.repo_id}
        metrics_desc = "; ".join(f"{m.name}={m.value}{m.unit}" for m in brain_graph.metrics)
        if metrics_desc:
            desc = desc + f" 关键指标：{metrics_desc}"
        self.upsert_concept_memory(
            concept_id=brain_concept_id,
            name=f"{brain_graph.repo_id} 脑图谱",
            description=desc,
            tags=tags,
            graph_ref={"repo_id": brain_graph.repo_id},
        )
        for region in brain_graph.regions.values():
            rid = ConceptId(f"region:{brain_graph.repo_id}:{region.id}")
            rdesc = f"类型={region.kind}，规模={region.size}"
            self.upsert_concept_memory(
                concept_id=rid,
                name=f"脑区:{region.name}",
                description=rdesc,
                tags={"brain_region", brain_graph.repo_id},
                graph_ref={"repo_id": brain_graph.repo_id, "region_id": region.id},
            )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from me_core.brain.graph import BrainGraph
