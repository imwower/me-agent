"""多模态感知（perception）相关桩实现。

当前提供：
- 若干可替换的 EncoderStub（Text/Image/Audio/Video）；
- encode_multimodal：接收 MultiModalInput，返回各模态的向量表示；
- BasePerception / TextPerception：感知接口与最简文本感知实现。
"""

from __future__ import annotations

import logging
from typing import Dict, List

from me_core.types import MultiModalInput

from .base import BasePerception, TextPerception, MultiModalPerception  # noqa: F401
from .audio_encoder_stub import AudioEncoderStub  # noqa: F401
from .image_encoder_stub import ImageEncoderStub  # noqa: F401
from .processor import encode_to_event  # noqa: F401
from .text_encoder_stub import TextEncoderStub  # noqa: F401
from .video_encoder_stub import VideoEncoderStub  # noqa: F401
from .image_perception import ImagePerception  # noqa: F401
from .audio_perception import AudioPerception  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "BasePerception",
    "TextPerception",
    "MultiModalPerception",
    "ImagePerception",
    "AudioPerception",
    "TextEncoderStub",
    "ImageEncoderStub",
    "AudioEncoderStub",
    "VideoEncoderStub",
    "encode_multimodal",
    "encode_to_event",
    "create_default_perception_pipeline",
]


def encode_multimodal(input_data: MultiModalInput) -> Dict[str, List[float]]:
    """对 MultiModalInput 中的各模态进行编码。

    返回：
        一个字典，键为模态名称（"text" / "image" / "audio" / "video"），
        值为对应的向量表示（定长 float 列表）。
    """

    logger.info("开始对多模态输入进行编码: %s", input_data)

    results: Dict[str, List[float]] = {}

    if input_data.text is not None:
        encoder = TextEncoderStub()
        results["text"] = encoder.encode(input_data.text)

    if input_data.image_meta is not None:
        encoder = ImageEncoderStub()
        results["image"] = encoder.encode(input_data.image_meta)

    if input_data.audio_meta is not None:
        encoder = AudioEncoderStub()
        results["audio"] = encoder.encode(input_data.audio_meta)

    if input_data.video_meta is not None:
        encoder = VideoEncoderStub()
        results["video"] = encoder.encode(input_data.video_meta)

    logger.info("多模态编码结果: %s", results)
    return results


def create_default_perception_pipeline() -> BasePerception:
    """构造一个默认的多模态感知流水线。

    当前实现：
        - 使用 MultiModalPerception，在传入 str 时退化为 TextPerception；
        - 后续可扩展为根据输入类型自动分派到 ImagePerception / AudioPerception 等。
    """

    return MultiModalPerception()
