from __future__ import annotations

import base64
import hashlib
import io
import logging
import math
import random
from typing import List, Optional

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - PIL 可能不存在
    Image = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch 可选
    torch = None  # type: ignore

from me_core.alignment.embeddings import EmbeddingBackend
from me_core.types import AudioRef, ImageRef

logger = logging.getLogger(__name__)


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _hash_vec(key: str, dim: int) -> List[float]:
    seed_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(seed_bytes, "big", signed=False)
    rng = random.Random(seed)
    return _normalize([rng.uniform(-1.0, 1.0) for _ in range(dim)])


class RealEmbeddingBackend(EmbeddingBackend):
    """
    真实多模态后端占位实现。

    - 若可用 torch + PIL，则尝试读取图片并生成基础像素直方图向量；
    - 文本用 hash 向量；音频暂时复用文本 hash。
    - 结构上与真实模型一致，便于后续替换为真正的 CLIP/多模态模型。
    """

    def __init__(self, model_config: Optional[dict] = None) -> None:
        self.model_config = model_config or {}
        self.dim = int(self.model_config.get("dim", 128))
        self.device = self.model_config.get("device", "cpu")
        # 预留真实模型加载位
        self.text_model = None
        self.image_model = None
        self.audio_model = None

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        result: List[List[float]] = []
        for t in texts:
            result.append(_hash_vec(t or "", self.dim))
        return result

    def _embed_image_pil(self, img: "Image.Image") -> List[float]:  # type: ignore[valid-type]
        # 将图像缩放到 8x8，作为直方图特征
        small = img.resize((8, 8)).convert("RGB")
        pixels = list(small.getdata())
        vec = []
        for r, g, b in pixels:
            vec.extend([r / 255.0, g / 255.0, b / 255.0])
        # 若向量短于 dim，用 hash 填充
        if len(vec) < self.dim:
            vec.extend(_hash_vec(str(len(vec)), self.dim - len(vec)))
        return _normalize(vec[: self.dim])

    def embed_image(self, image_refs: List[ImageRef]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for ref in image_refs:
            path = ref.path
            if Image is None:
                vectors.append(_hash_vec(path, self.dim))
                continue
            try:
                with open(path, "rb") as f:
                    data = f.read()
                img = Image.open(io.BytesIO(data))
                vec = self._embed_image_pil(img)
            except Exception:
                vec = _hash_vec(path, self.dim)
            vectors.append(vec)
        return vectors

    def embed_audio(self, audio_refs: List[AudioRef]) -> List[List[float]]:
        # 预留音频处理接口，当前复用 hash 占位
        return [_hash_vec(ref.path, self.dim) for ref in audio_refs]


def create_backend(config_dict: dict) -> EmbeddingBackend:
    """
    供 me_core.alignment.create_embedding_backend_from_config 使用。
    """

    return RealEmbeddingBackend(model_config=config_dict or {})
