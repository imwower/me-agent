from __future__ import annotations

import logging
from typing import Any, Dict, List

from me_core.types import AgentEvent, MultiModalInput

from .audio_encoder_stub import AudioEncoderStub
from .image_encoder_stub import ImageEncoderStub
from .text_encoder_stub import TextEncoderStub
from .video_encoder_stub import VideoEncoderStub

logger = logging.getLogger(__name__)


def _encode_multimodal_local(input_data: MultiModalInput) -> Dict[str, List[float]]:
    """局部实现的多模态编码逻辑，避免与包级别导入产生循环依赖。"""

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

    return results


def encode_to_event(
    input_data: MultiModalInput,
    source: str = "perception",
) -> AgentEvent:
    """将多模态输入编码为向量后，封装成一个 AgentEvent。

    事件结构示例（payload）：
        {
            "kind": "perception",
            "source": "perception" 或其他调用来源标记,
            "modalities": ["text", "image", ...],
            "embeddings": {"text": [...], "image": [...]},
            "raw": {
                "text": "...",
                "image_meta": {...},
                "audio_meta": {...},
                "video_meta": {...},
            },
        }
    """

    embeddings = _encode_multimodal_local(input_data)

    payload: Dict[str, Any] = {
        "kind": "perception",
        "source": source,
        "modalities": list(embeddings.keys()),
        "embeddings": embeddings,
        "raw": {
            "text": input_data.text,
            "image_meta": input_data.image_meta,
            "audio_meta": input_data.audio_meta,
            "video_meta": input_data.video_meta,
        },
    }

    logger.info(
        "将多模态输入封装为感知事件: source=%s, modalities=%s",
        source,
        payload["modalities"],
    )

    return AgentEvent.now(event_type="perception", payload=payload)
