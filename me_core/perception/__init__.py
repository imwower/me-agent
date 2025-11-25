"""多模态感知（perception）相关桩实现。

当前提供：
- 文本/图片/音频感知器（TextPerception/ImagePerception/AudioPerception）；
- MultiModalPerception 兼容旧接口；
- encode_multimodal：接收 MultiModalInput，返回各模态的伪向量表示；
- default_perceive：根据输入类型自动选择合适的感知器。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List
import json

from me_core.types import AudioRef, ImageRef, MultiModalInput, AgentEvent, VideoRef

from .audio_perception import AudioPerception  # noqa: F401
from .audio_encoder_stub import AudioEncoderStub  # noqa: F401
from .base import BasePerception  # noqa: F401
from .image_encoder_stub import ImageEncoderStub  # noqa: F401
from .image_perception import ImagePerception  # noqa: F401
from .multimodal_perception import MultiModalPerception  # noqa: F401
from .structured_perception import StructuredPerception  # noqa: F401
from .processor import encode_to_event  # noqa: F401
from .text_encoder_stub import TextEncoderStub  # noqa: F401
from .text_perception import TextPerception  # noqa: F401
from .video_encoder_stub import VideoEncoderStub  # noqa: F401
from .video_perception import VideoPerception  # noqa: F401

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
    "StructuredPerception",
    "VideoPerception",
    "encode_multimodal",
    "encode_to_event",
    "default_perceive",
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

    if input_data.structured_data is not None:
        encoder = TextEncoderStub()
        try:
            encoded = encoder.encode(json.dumps(input_data.structured_data, sort_keys=True))
        except Exception:
            encoded = encoder.encode(str(input_data.structured_data))
        results["structured"] = encoded

    logger.info("多模态编码结果: %s", results)
    return results


def default_perceive(raw_input: Any) -> List[AgentEvent]:
    """
    根据 raw_input 类型分派到合适的感知器：
    - str 或 list[str]: 当作文本
    - ImageRef 或常见图片后缀的路径: 当作图片
    - AudioRef: 当作音频
    - MultiModalInput: 使用 MultiModalPerception 兼容旧逻辑
    """

    if isinstance(raw_input, MultiModalInput):
        return MultiModalPerception().perceive(raw_input)

    if isinstance(raw_input, (str, list)):
        # 如果是字符串路径，demo 可自行传给 ImagePerception，这里默认视为文本
        try:
            suffix = Path(str(raw_input)).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                return ImagePerception().perceive(str(raw_input))
            if suffix in {".wav", ".mp3", ".flac", ".aac"}:
                return AudioPerception().perceive(str(raw_input))
            if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
                return VideoPerception().perceive(str(raw_input))
        except Exception:
            pass
        return TextPerception().perceive(raw_input)

    if isinstance(raw_input, ImageRef):
        return ImagePerception().perceive(raw_input)

    if isinstance(raw_input, AudioRef):
        return AudioPerception().perceive(raw_input)

    if isinstance(raw_input, VideoRef):
        return VideoPerception().perceive(raw_input)

    if isinstance(raw_input, dict):
        # 偏向结构化感知
        return StructuredPerception().perceive(raw_input)

    if isinstance(raw_input, str):
        try:
            obj = json.loads(raw_input)
            if isinstance(obj, dict):
                return StructuredPerception().perceive(obj)
        except Exception:
            pass
        suffix = Path(raw_input).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            return ImagePerception().perceive(raw_input)
        if suffix in {".wav", ".mp3", ".flac", ".aac"}:
            return AudioPerception().perceive(raw_input)
        if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
            return VideoPerception().perceive(raw_input)
        return TextPerception().perceive(raw_input)

    return []
