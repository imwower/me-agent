from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..types import AgentEvent, ImageRef, AudioRef
from .concepts import ConceptSpace, ConceptNode, _cosine_similarity
from .embeddings import EmbeddingBackend, DummyEmbeddingBackend


@dataclass
class MultimodalAligner:
    backend: EmbeddingBackend
    concept_space: ConceptSpace
    similarity_threshold: float = 0.8

    @classmethod
    def with_dummy_backend(cls, similarity_threshold: float = 0.8) -> "MultimodalAligner":
        space = ConceptSpace()
        backend = DummyEmbeddingBackend()
        return cls(backend=backend, concept_space=space, similarity_threshold=similarity_threshold)

    def _extract_text(self, event: AgentEvent) -> str:
        payload = event.payload or {}
        if isinstance(payload, dict):
            if isinstance(payload.get("text"), str):
                return payload["text"]
            raw = payload.get("raw")
            if isinstance(raw, dict) and isinstance(raw.get("text"), str):
                return raw["text"]
        return ""

    def _extract_image_ref(self, payload: object) -> Optional[ImageRef]:
        if isinstance(payload, dict):
            ref = payload.get("image_ref")
            if isinstance(ref, ImageRef):
                return ref
            if isinstance(payload.get("path"), str):
                return ImageRef(path=str(payload.get("path")))
        return None

    def _extract_audio_ref(self, payload: object) -> Optional[AudioRef]:
        if isinstance(payload, dict):
            ref = payload.get("audio_ref")
            if isinstance(ref, AudioRef):
                return ref
            if isinstance(payload.get("path"), str):
                return AudioRef(path=str(payload.get("path")))
        return None

    def align_event(self, event: AgentEvent) -> Optional[ConceptNode]:
        """
        根据事件的模态与 payload，生成 embedding，并在 ConceptSpace 中找到/创建对应概念。
        将 embedding 填回 event.embedding。
        返回对应的 ConceptNode（或 None）。
        """

        payload = event.payload or {}
        modality = event.modality or payload.get("modality") or "text"
        event.modality = modality

        embedding = None
        name_hint: Optional[str] = None

        if modality == "text":
            text = self._extract_text(event)
            if text:
                embedding = self.backend.embed_text([text])[0]
                name_hint = text[:16]
        elif modality == "image":
            image_ref = self._extract_image_ref(payload)
            if image_ref is not None:
                embedding = self.backend.embed_image([image_ref])[0]
                name_hint = Path(image_ref.path).stem
        elif modality == "audio":
            audio_ref = self._extract_audio_ref(payload)
            if audio_ref is not None:
                embedding = self.backend.embed_audio([audio_ref])[0]
                name_hint = Path(audio_ref.path).stem

        if embedding is None:
            text = self._extract_text(event)
            if text:
                embedding = self.backend.embed_text([text])[0]
                name_hint = name_hint or text[:16]

        if embedding is None:
            return None

        concept = self.concept_space.get_or_create(embedding, name_hint, self.similarity_threshold)
        self.concept_space.link_observation(
            concept,
            {"modality": modality, "event_id": event.id, "payload_keys": list(payload.keys()) if isinstance(payload, dict) else []},
            embedding,
        )
        event.embedding = list(embedding)
        event.meta.setdefault("concept_id", str(concept.id))
        return concept

    def align_pair(self, text: str, image_ref: ImageRef) -> tuple[ConceptNode, float, float]:
        """
        用于 demo/testing：对齐一条文本 + 一张图片，返回概念和双方相似度信息。
        """

        text_vec = self.backend.embed_text([text])[0]
        image_vec = self.backend.embed_image([image_ref])[0]
        concept = self.concept_space.get_or_create(text_vec, text[:16], self.similarity_threshold)
        self.concept_space.link_observation(concept, {"modality": "text"}, text_vec)
        self.concept_space.link_observation(concept, {"modality": "image"}, image_vec)

        sim_text = _cosine_similarity(text_vec, concept.centroid)
        sim_image = _cosine_similarity(image_vec, concept.centroid)
        return concept, float(sim_text), float(sim_image)


__all__ = ["MultimodalAligner"]
