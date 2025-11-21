from __future__ import annotations

import hashlib
import math
import random
from typing import List, Protocol

from ..types import AudioRef, ImageRef


class EmbeddingBackend(Protocol):
    def embed_text(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_image(self, image_refs: List[ImageRef]) -> List[List[float]]:
        ...

    def embed_audio(self, audio_refs: List[AudioRef]) -> List[List[float]]:
        ...


class DummyEmbeddingBackend:
    """
    R0 版：使用 hash + 伪随机的方式生成“稳定但无语义”的假向量。
    - 同样的文本/路径 -> 同一个向量
    - 不同文本/路径 -> 向量足够分散
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _hash_to_vector(self, key: str) -> List[float]:
        seed_bytes = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(seed_bytes, "big", signed=False)
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_image(self, image_refs: List[ImageRef]) -> List[List[float]]:
        return [self._hash_to_vector(ref.path) for ref in image_refs]

    def embed_audio(self, audio_refs: List[AudioRef]) -> List[List[float]]:
        return [self._hash_to_vector(ref.path) for ref in audio_refs]


__all__ = ["EmbeddingBackend", "DummyEmbeddingBackend"]
