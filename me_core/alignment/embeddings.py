from __future__ import annotations

import hashlib
import math
import random
import importlib
from typing import List, Protocol

try:  # 避免在 me_core 内部强依赖配置模块
    from me_core.config import AgentConfig  # type: ignore
except Exception:  # pragma: no cover - 仅用于静态检查
    AgentConfig = None  # type: ignore

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


def create_embedding_backend_from_config(config: AgentConfig | None = None) -> EmbeddingBackend:
    """
    根据配置创建 embedding backend。
    默认使用 DummyEmbeddingBackend，若配置要求非 dummy，则尝试从扩展模块加载。
    """

    use_dummy = True
    module_name: str | None = None
    factory_name = "create_backend"
    if config is not None:
        use_dummy = bool(getattr(config, "use_dummy_embedding", True))
        module_name = getattr(config, "embedding_backend_module", None)

    if use_dummy:
        return DummyEmbeddingBackend()

    if module_name:
        try:
            module = importlib.import_module(str(module_name))
            factory = getattr(module, factory_name, None)
            if callable(factory):
                backend = factory(config)  # type: ignore[misc]
                if backend is not None:
                    return backend
        except Exception:
            pass

    # 回退到 Dummy，保证核心流程不中断
    return DummyEmbeddingBackend()


# 兼容旧接口命名
def create_embedding_backend(config: AgentConfig | None = None) -> EmbeddingBackend:
    return create_embedding_backend_from_config(config)


__all__ = [
    "EmbeddingBackend",
    "DummyEmbeddingBackend",
    "create_embedding_backend_from_config",
    "create_embedding_backend",
]
