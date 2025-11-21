from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Set, Tuple

ConceptId = NewType("ConceptId", str)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass
class ConceptNode:
    id: ConceptId
    name: str
    aliases: Set[str] = field(default_factory=set)
    centroid: List[float] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


class ConceptSpace:
    """
    简单的概念空间，保存所有概念节点，并提供最近邻检索和观测更新。
    """

    def __init__(self) -> None:
        self._concepts: List[ConceptNode] = []

    def add_concept(self, name: str, embedding: List[float]) -> ConceptNode:
        node = ConceptNode(
            id=ConceptId(str(uuid.uuid4())),
            name=name,
            aliases={name},
            centroid=list(embedding),
        )
        self._concepts.append(node)
        return node

    def all_concepts(self) -> List[ConceptNode]:
        return list(self._concepts)

    def find_nearest(self, embedding: List[float], top_k: int = 5) -> List[Tuple[ConceptNode, float]]:
        """
        使用简单余弦相似度（用标准库 math 实现）做线性扫描。
        """

        sims: List[Tuple[ConceptNode, float]] = []
        for node in self._concepts:
            sim = _cosine_similarity(embedding, node.centroid)
            sims.append((node, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def get_or_create(self, embedding: List[float], name_hint: str | None, threshold: float = 0.8) -> ConceptNode:
        """
        若最近概念相似度 >= threshold，则返回该概念，否则创建新概念。
        """

        nearest = self.find_nearest(embedding, top_k=1)
        if nearest and nearest[0][1] >= threshold:
            return nearest[0][0]

        name = name_hint.strip() if isinstance(name_hint, str) and name_hint.strip() else f"concept_{len(self._concepts) + 1}"
        return self.add_concept(name, embedding)

    def link_observation(self, concept: ConceptNode, event_info: Dict[str, Any], embedding: List[float]) -> None:
        """
        将一次观测追加到 concept.examples，并更新 centroid（简单平均）。
        """

        concept.examples.append({"info": event_info, "embedding": list(embedding)})
        n = len(concept.examples)
        if concept.centroid:
            concept.centroid = [
                ((n - 1) * c + e) / float(n) for c, e in zip(concept.centroid, embedding)
            ]
        else:
            concept.centroid = list(embedding)


__all__ = [
    "ConceptId",
    "ConceptNode",
    "ConceptSpace",
]
