from __future__ import annotations

import hashlib
import logging
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


def _hash_to_vector(payload: str, dim: int = 16) -> List[float]:
    """将字符串通过哈希映射为定长向量（图像专用桩）。"""

    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    if len(digest) < dim:
        repeat = (dim + len(digest) - 1) // len(digest)
        digest = (digest * repeat)[:dim]
    else:
        digest = digest[:dim]
    return [b / 255.0 for b in digest]


class ImageEncoderStub:
    """图像编码桩实现。

    将图像的元信息（如文件名、标签等）映射为定长向量。
    """

    def encode(self, *args: Any, **kwargs: Any) -> List[float]:
        """对图像相关输入进行编码。"""

        payload_tuple: Tuple[Any, ...] = args
        kwargs_items = tuple(sorted(kwargs.items()))
        payload_str = f"image:{repr((payload_tuple, kwargs_items))}"

        logger.info("图像编码输入: args=%s, kwargs=%s", args, kwargs)
        vector = _hash_to_vector(payload_str)
        logger.info("图像编码结果向量长度=%d", len(vector))
        return vector

