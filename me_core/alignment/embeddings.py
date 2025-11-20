from __future__ import annotations

import hashlib
import math
import logging
from dataclasses import dataclass
from typing import List, Protocol, Optional

logger = logging.getLogger(__name__)

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


# --------------------------------------------------------------------------- #
# 可选的真实多模态后端（基于 torchvision）
# --------------------------------------------------------------------------- #


@dataclass
class TorchVisionEmbeddingBackend:
    """使用 torchvision 预训练模型生成更真实的图像向量。

    设计：
        - 文本仍使用 hash 方案（ lightweight ），确保无额外依赖；
        - 图像使用 resnet18 预训练特征（池化后向量长度 512）；
        - 模型懒加载，仅在首次调用 embed_image 时才会加载权重。
    """

    device: str = "cpu"
    use_pretrained: bool = True
    _model: Optional[object] = None
    _transform: Optional[object] = None
    _dim: int = 512

    def _lazy_init(self) -> None:
        """延迟加载 torchvision 模型与预处理。"""

        if self._model is not None:
            return

        try:  # 局部导入，避免在未安装 torchvision 时出错
            import torch  # type: ignore
            import torchvision  # type: ignore
            from torchvision import transforms  # type: ignore
            from torchvision.models import resnet18  # type: ignore
            from torchvision.models import ResNet18_Weights  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "未安装 torch/torchvision，无法使用 TorchVisionEmbeddingBackend。"
            ) from exc

        weights = ResNet18_Weights.DEFAULT if self.use_pretrained else None
        model = resnet18(weights=weights)
        model.eval()
        # 去掉最后的分类层，保留 avgpool 输出
        model = torch.nn.Sequential(*(list(model.children())[:-1]))  # type: ignore[attr-defined]
        self._model = model.to(self.device)
        self._transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._dim = 512  # resnet18 avgpool 输出通道
        logger.info("已加载 torchvision resnet18 模型用于多模态嵌入。")

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        # 文本仍使用与 Dummy 相同的 hash 方案，保证依赖轻量
        return [_hash_to_unit_vector("text", t, dim=64) for t in texts]

    def embed_image(self, image_refs: List[ImageRef]) -> List[List[float]]:
        self._lazy_init()
        import torch  # type: ignore
        from PIL import Image  # type: ignore

        model = self._model  # type: ignore[assignment]
        transform = self._transform  # type: ignore[assignment]
        assert model is not None and transform is not None

        vectors: List[List[float]] = []
        with torch.no_grad():
            for ref in image_refs:
                try:
                    img = Image.open(ref.path).convert("RGB")
                    tensor = transform(img).unsqueeze(0).to(self.device)
                    feat = model(tensor)  # shape (1, 512, 1, 1)
                    feat = feat.flatten().cpu().numpy().astype(float)
                    vec = feat.tolist()
                    # 单位化，消除尺度差异
                    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
                    vec = [x / norm for x in vec]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("读取/编码图像失败，将使用占位向量: %s", exc)
                    vec = _hash_to_unit_vector("image", ref.path, dim=self._dim)
                vectors.append(vec)
        return vectors

    def embed_audio(self, audio_refs: List[AudioRef]) -> List[List[float]]:
        # 仍然使用 hash 占位，以保持依赖最小化
        return [_hash_to_unit_vector("audio", ref.path, dim=64) for ref in audio_refs]


__all__.append("TorchVisionEmbeddingBackend")
