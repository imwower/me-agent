from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, NewType, Optional, Tuple

from me_core.types import AgentEvent

ConceptId = NewType("ConceptId", str)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass(slots=True)
class ConceptNode:
    """表示概念空间中的一个概念节点。

    字段说明：
        id: 概念唯一标识；
        name: 概念名称（中文/英文均可）；
        aliases: 同义词或别名，用于文本检索；
        centroid: 概念中心向量（多模态共享表示）；
        examples: 历史样本列表，每条包含 modality/embedding/payload_ref 等；
        meta: 其他元信息，例如创建时间、手工备注等。
    """

    id: ConceptId
    name: str
    centroid: List[float]
    aliases: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """将 ConceptNode 转换为可序列化字典。"""

        return {
            "id": str(self.id),
            "name": self.name,
            "aliases": list(self.aliases),
            "centroid": list(self.centroid),
            "examples": list(self.examples),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptNode":
        """从字典恢复 ConceptNode。"""

        return cls(
            id=ConceptId(str(data.get("id") or "")),
            name=str(data.get("name") or ""),
            centroid=list(data.get("centroid") or []),
            aliases=list(data.get("aliases") or []),
            examples=list(data.get("examples") or []),
            meta=dict(data.get("meta") or {}),
        )


class ConceptSpace:
    """简单的概念空间实现。

    设计目标：
        - 仅依赖标准库，使用线性扫描实现最近邻搜索；
        - 为多模态对齐器提供“查找或创建概念”的基础能力；
        - 后续可替换为更高效的索引结构，而不影响上层接口。
    """

    def __init__(self, similarity_threshold: float = 0.6) -> None:
        self.similarity_threshold = similarity_threshold
        self._nodes: Dict[ConceptId, ConceptNode] = {}
        self._order: List[ConceptId] = []
        self._alias_index: Dict[str, ConceptId] = {}

    # 基础操作 ---------------------------------------------------------------------

    def all_concepts(self) -> List[ConceptNode]:
        return [self._nodes[cid] for cid in self._order]

    def get(self, concept_id: ConceptId) -> Optional[ConceptNode]:
        return self._nodes.get(concept_id)

    # 检索与增量更新 ---------------------------------------------------------------

    def find_nearest(
        self,
        embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[ConceptNode, float]]:
        """按余弦相似度返回最近的若干概念。"""

        sims: List[Tuple[ConceptNode, float]] = []
        for cid in self._order:
            node = self._nodes[cid]
            sim = _cosine_similarity(embedding, node.centroid)
            sims.append((node, sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def get_or_create(
        self,
        embedding: List[float],
        name_hint: Optional[str] = None,
    ) -> ConceptNode:
        """根据向量在空间中查找最近概念，若相似度不足则创建新概念。"""

        best: Optional[Tuple[ConceptNode, float]] = None
        for node, sim in self.find_nearest(embedding, top_k=1):
            best = (node, sim)
            break

        if best is not None and best[1] >= self.similarity_threshold:
            return best[0]

        # 创建新概念
        cid = ConceptId(str(uuid.uuid4()))
        name = name_hint or f"concept_{len(self._order) + 1}"
        node = ConceptNode(
            id=cid,
            name=name,
            centroid=list(embedding),
            aliases=[name],
        )
        self._nodes[cid] = node
        self._order.append(cid)
        for alias in node.aliases:
            key = alias.strip().lower()
            if key:
                self._alias_index.setdefault(key, cid)
        return node

    def link_observation(
        self,
        event: AgentEvent,
        embedding: List[float],
        *,
        name_hint: Optional[str] = None,
    ) -> ConceptNode:
        """将一次观测样本链接到概念空间中。

        行为：
            - 根据 embedding 在概念空间中查找/创建概念；
            - 将该样本追加到 ConceptNode.examples；
            - 更新 AgentEvent.embedding 字段。
        """

        node = self.get_or_create(embedding, name_hint=name_hint)

        example = {
            "event_id": event.id,
            "modality": event.modality,
            "embedding": list(embedding),
        }
        node.examples.append(example)

        # 简单更新中心向量（增量平均），避免在小样本时过度震荡
        n = len(node.examples)
        alpha = 1.0 / float(n)
        node.centroid = [
            (1.0 - alpha) * c + alpha * e for c, e in zip(node.centroid, embedding)
        ]

        event.embedding = list(embedding)
        event.meta.setdefault("concept_id", str(node.id))
        return node

    # 文本友好的检索接口 ---------------------------------------------------------

    def find_by_alias(self, name: str) -> Optional[ConceptNode]:
        key = name.strip().lower()
        cid = self._alias_index.get(key)
        return self._nodes.get(cid) if cid is not None else None

    # 序列化 / 反序列化 -----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """将整个概念空间转换为可序列化字典。"""

        return {
            "similarity_threshold": self.similarity_threshold,
            "concepts": [node.to_dict() for node in self.all_concepts()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConceptSpace":
        """从字典构造 ConceptSpace 实例。"""

        space = cls(similarity_threshold=float(data.get("similarity_threshold", 0.6)))
        concepts_raw = data.get("concepts") or []
        for item in concepts_raw:
            node = ConceptNode.from_dict(item)
            cid = node.id
            space._nodes[cid] = node
            space._order.append(cid)
            for alias in node.aliases:
                key = alias.strip().lower()
                if key:
                    space._alias_index.setdefault(key, cid)
        return space

    def register_alias(self, node: ConceptNode, alias: str) -> None:
        """为指定概念注册一个新的别名，并更新别名索引。"""

        alias = alias.strip()
        if not alias:
            return
        if alias not in node.aliases:
            node.aliases.append(alias)
        key = alias.lower()
        self._alias_index.setdefault(key, node.id)


__all__ = ["ConceptId", "ConceptNode", "ConceptSpace"]
