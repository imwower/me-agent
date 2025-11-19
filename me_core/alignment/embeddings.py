from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Protocol

from me_core.types import AudioRef, ImageRef


class EmbeddingBackend(Protocol):
    """多模态嵌入后端接口。

    设计目标：
        - 在 me_core 中只依赖轻量接口，不绑定具体深度学习框架；
        - 真实模型（如 CLIP、多模态 LLM）可以在独立扩展包中实现该接口；
        - 默认实现 DummyEmbeddingBackend 使用 hash 生成可重复伪向量，便于开发与测试。
    """

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_image(self, image_refs: List[ImageRef]) -> List[List[float]]:
        ...

    def embed_audio(self, audio_refs: List[AudioRef]) -> List[List[float]]:
        ...


def _hash_to_unit_vector(namespace: str, key: str, dim: int = 32) -> List[float]:
    """将字符串映射为定长单位向量（伪随机但可重复）。"""

    h = hashlib.sha256(f"{namespace}::{key}".encode("utf-8")).digest()
    # 将 hash bytes 切片映射为 dim 维向量
    vals = [int.from_bytes(h[i : i + 2], "big", signed=False) for i in range(0, 2 * dim, 2)]
    # 归一化为 0-1，再中心化到 [-0.5, 0.5]
    floats = [(v / 65535.0) - 0.5 for v in vals]
    # L2 归一化到单位长度，避免不同 key 的向量长度差异影响相似度
    norm = math.sqrt(sum(x * x for x in floats)) or 1.0
    return [x / norm for x in floats]


@dataclass
class DummyEmbeddingBackend:
    """默认的伪嵌入后端实现。

    仅依赖 hashlib 与数学运算：
        - 对于文本：使用原始字符串作为 key；
        - 对于图像：使用 path + 基本元信息；
        - 对于音频：使用 path + duration + sample_rate。
    """

    dim: int = 32

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        return [_hash_to_unit_vector("text", t, self.dim) for t in texts]

    def embed_image(self, image_refs: List[ImageRef]) -> List[List[float]]:
        keys: List[str] = []
        for ref in image_refs:
            key = f"{ref.path}|{ref.width or 0}x{ref.height or 0}"
            keys.append(key)
        return [_hash_to_unit_vector("image", k, self.dim) for k in keys]

    def embed_audio(self, audio_refs: List[AudioRef]) -> List[List[float]]:
        keys: List[str] = []
        for ref in audio_refs:
            key = f"{ref.path}|{ref.duration or 0.0}|{ref.sample_rate or 0}"
            keys.append(key)
        return [_hash_to_unit_vector("audio", k, self.dim) for k in keys]


__all__ = ["EmbeddingBackend", "DummyEmbeddingBackend"]

