from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from me_core.types import AgentEvent, AudioRef, ImageRef

from .concepts import ConceptNode, ConceptSpace
from .embeddings import DummyEmbeddingBackend, EmbeddingBackend, TorchVisionEmbeddingBackend


@dataclass
class MultimodalAligner:
    """多模态对齐器（最小可运行版）。

    职责：
        - 基于文本/图像/音频内容生成嵌入；
        - 将观测事件对齐到概念空间中的某个 ConceptNode；
        - 在事件上回填 embedding 字段，便于后续模块使用。
    """

    backend: EmbeddingBackend
    concept_space: ConceptSpace

    @classmethod
    def with_dummy_backend(cls, similarity_threshold: float = 0.6) -> "MultimodalAligner":
        """构造一个使用 DummyEmbeddingBackend + 默认 ConceptSpace 的对齐器。"""

        space = ConceptSpace(similarity_threshold=similarity_threshold)
        backend = DummyEmbeddingBackend()
        return cls(backend=backend, concept_space=space)

    @classmethod
    def with_torchvision_backend(
        cls,
        similarity_threshold: float = 0.6,
        device: str = "cpu",
        use_pretrained: bool = True,
    ) -> "MultimodalAligner":
        """构造一个使用 TorchVisionEmbeddingBackend 的对齐器。"""

        space = ConceptSpace(similarity_threshold=similarity_threshold)
        backend = TorchVisionEmbeddingBackend(device=device, use_pretrained=use_pretrained)
        return cls(backend=backend, concept_space=space)

    @classmethod
    def with_auto_backend(cls, similarity_threshold: float = 0.6) -> "MultimodalAligner":
        """优先尝试真实嵌入后端，失败时回退到 Dummy。"""

        try:
            return cls.with_torchvision_backend(similarity_threshold=similarity_threshold)
        except Exception:  # noqa: BLE001
            return cls.with_dummy_backend(similarity_threshold=similarity_threshold)

    # --------------------------------------------------------------------- #
    # 对齐主入口
    # --------------------------------------------------------------------- #

    def align_event(self, event: AgentEvent) -> Optional[ConceptNode]:
        """对单个事件执行多模态对齐。

        当前策略：
            - 若有文本，则优先使用文本作为对齐依据；
            - 否则若有图像元信息，则使用图像；
            - 否则若有音频元信息，则使用音频；
            - 若无法提取有效内容，则返回 None。
        """

        payload = event.payload or {}
        raw = payload.get("raw") if isinstance(payload, dict) else None

        text: Optional[str] = None
        image_ref: Optional[ImageRef] = None
        audio_ref: Optional[AudioRef] = None

        if isinstance(raw, dict):
            if isinstance(raw.get("text"), str):
                text = raw["text"]
            img_meta = raw.get("image_meta")
            if isinstance(img_meta, dict) and img_meta.get("path"):
                image_ref = ImageRef(
                    path=str(img_meta.get("path")),
                    width=img_meta.get("width"),
                    height=img_meta.get("height"),
                    meta={k: v for k, v in img_meta.items() if k not in {"path", "width", "height"}},
                )
            audio_meta = raw.get("audio_meta")
            if isinstance(audio_meta, dict) and audio_meta.get("path"):
                audio_ref = AudioRef(
                    path=str(audio_meta.get("path")),
                    duration=audio_meta.get("duration"),
                    sample_rate=audio_meta.get("sample_rate"),
                    meta={k: v for k, v in audio_meta.items() if k not in {"path", "duration", "sample_rate"}},
                )

        # 按优先级选择模态
        embedding = None
        name_hint: Optional[str] = None

        if text:
            embedding = self.backend.embed_text([text])[0]
            name_hint = text[:16]
            event.modality = event.modality or "text"
        elif image_ref is not None:
            embedding = self.backend.embed_image([image_ref])[0]
            name_hint = image_ref.path.split("/")[-1]
            event.modality = event.modality or "image"
        elif audio_ref is not None:
            embedding = self.backend.embed_audio([audio_ref])[0]
            name_hint = audio_ref.path.split("/")[-1]
            event.modality = event.modality or "audio"

        if embedding is None:
            return None

        node = self.concept_space.link_observation(
            event=event,
            embedding=embedding,
            name_hint=name_hint,
        )
        return node

    # --------------------------------------------------------------------- #
    # 文本-图像对齐辅助函数
    # --------------------------------------------------------------------- #

    def align_pair(
        self,
        text: str,
        image_ref: ImageRef,
    ) -> tuple[ConceptNode, float, float]:
        """对齐一对文本-图像样本，并返回概念及两种模态的相似度估计。

        返回：
            (concept, sim_text, sim_image)
        """

        text_embedding = self.backend.embed_text([text])[0]
        image_embedding = self.backend.embed_image([image_ref])[0]

        # 使用文本 embedding 作为概念锚点
        dummy_event = AgentEvent.now(
            event_type="alignment_pair",
            payload={
                "raw": {
                    "text": text,
                    "image_meta": {
                        "path": image_ref.path,
                        "width": image_ref.width,
                        "height": image_ref.height,
                        "meta": dict(image_ref.meta),
                    },
                }
            },
            source="alignment",
        )

        concept = self.concept_space.link_observation(
            event=dummy_event,
            embedding=text_embedding,
            name_hint=text[:16],
        )

        # 计算图像向量与概念中心的相似度
        from .concepts import _cosine_similarity  # 局部导入避免循环

        sim_text = _cosine_similarity(text_embedding, concept.centroid)
        sim_image = _cosine_similarity(image_embedding, concept.centroid)
        return concept, float(sim_text), float(sim_image)


__all__ = ["MultimodalAligner"]
