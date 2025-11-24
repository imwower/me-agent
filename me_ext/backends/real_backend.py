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
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
except Exception:  # pragma: no cover - torch/transformers 可选
    torch = None  # type: ignore
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore

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
    真实多模态后端。

    默认尝试加载 transformers CLIP（如 openai/clip-vit-base-patch32），
    若初始化失败则回退到 hash 占位向量，保证流程可用。
    """

    def __init__(self, model_config: Optional[dict] = None) -> None:
        self.model_config = model_config or {}
        self.dim = int(self.model_config.get("dim", 512))
        self.device = str(self.model_config.get("device", "cpu"))
        self.max_batch_size = int(self.model_config.get("max_batch_size", 8))
        self.model_name = self.model_config.get("model_name_or_path", "openai/clip-vit-base-patch32")
        self.use_stub = bool(self.model_config.get("use_stub", False))
        self.text_model = None
        self.image_model = None
        self.audio_model = None
        self.processor = None
        self._init_model()

    def _init_model(self) -> None:
        if self.use_stub:
            logger.info("RealEmbeddingBackend 处于 stub 模式，仅使用 hash 向量。")
            return
        if torch is None or CLIPModel is None or CLIPProcessor is None:
            logger.warning("缺少 torch/transformers，RealEmbeddingBackend 将退回 hash 向量。")
            return
        try:
            device = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
            self.device = str(device)
            self.model = CLIPModel.from_pretrained(self.model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.dim = int(self.model.config.projection_dim)
            logger.info("加载 CLIP 模型成功: %s on %s", self.model_name, self.device)
        except Exception as exc:  # pragma: no cover - 依赖外部模型
            logger.warning("加载 CLIP 模型失败，回退到 hash 后端: %s", exc)
            self.model = None
            self.processor = None

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if getattr(self, "model", None) is None or self.processor is None or torch is None or self.use_stub:
            return [_hash_vec(t or "", self.dim) for t in texts]
        vectors: List[List[float]] = []
        self.model.eval()
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            try:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    feats = self.model.get_text_features(**inputs)
                feats = torch.nn.functional.normalize(feats, dim=-1)
                vectors.extend(feats.detach().cpu().tolist())
            except Exception as exc:  # pragma: no cover - 依赖外部模型
                logger.warning("文本编码失败，回退 hash: %s", exc)
                vectors.extend(_hash_vec(t or "", self.dim) for t in batch)
        return vectors

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
        if not image_refs:
            return []
        if getattr(self, "model", None) is None or self.processor is None or torch is None or self.use_stub:
            return [_hash_vec(ref.path, self.dim) for ref in image_refs]
        if Image is None:
            logger.warning("PIL 不可用，图片编码回退 hash。")
            return [_hash_vec(ref.path, self.dim) for ref in image_refs]
        vectors: List[List[float]] = []
        imgs: List["Image.Image"] = []  # type: ignore[valid-type]
        for ref in image_refs:
            try:
                with open(ref.path, "rb") as f:
                    data = f.read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
                imgs.append(img)
            except Exception:
                imgs.append(None)  # type: ignore[arg-type]
        for i in range(0, len(imgs), self.max_batch_size):
            batch_imgs = imgs[i : i + self.max_batch_size]
            valid_imgs = [im for im in batch_imgs if im is not None]
            if not valid_imgs:
                vectors.extend(_hash_vec(image_refs[j].path, self.dim) for j in range(i, i + len(batch_imgs)))
                continue
            inputs = self.processor(images=valid_imgs, return_tensors="pt")
            try:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    feats = self.model.get_image_features(**inputs)
                feats = torch.nn.functional.normalize(feats, dim=-1).detach().cpu().tolist()
            except Exception as exc:  # pragma: no cover
                logger.warning("图片编码失败，回退 hash: %s", exc)
                feats = []
            # 将编码结果与原顺序对齐，不足部分填 hash
            feat_iter = iter(feats)
            for idx, img in enumerate(batch_imgs):
                if img is None:
                    vectors.append(_hash_vec(image_refs[i + idx].path, self.dim))
                else:
                    vectors.append(next(feat_iter, _hash_vec(image_refs[i + idx].path, self.dim)))
        return vectors

    def embed_audio(self, audio_refs: List[AudioRef]) -> List[List[float]]:
        # 预留音频处理接口，当前复用 hash 占位
        return [_hash_vec(ref.path, self.dim) for ref in audio_refs]


def create_backend(config_dict: dict) -> EmbeddingBackend:
    """
    供 me_core.alignment.create_embedding_backend_from_config 使用。
    """

    return RealEmbeddingBackend(model_config=config_dict or {})
