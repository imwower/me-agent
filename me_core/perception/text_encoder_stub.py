from __future__ import annotations

import hashlib
import logging
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


def _hash_to_vector(payload: str, dim: int = 16) -> List[float]:
    """将字符串通过哈希映射为定长向量。

    实现约定：
        - 使用 SHA256 生成稳定哈希；
        - 取前 dim 字节，并映射到 [0,1] 的浮点数；
        - 同一 payload 始终对应相同向量。
    """

    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    if len(digest) < dim:
        # 若字节数不足，则重复拼接后再截断
        repeat = (dim + len(digest) - 1) // len(digest)
        digest = (digest * repeat)[:dim]
    else:
        digest = digest[:dim]
    return [b / 255.0 for b in digest]


class TextEncoderStub:
    """文本编码桩实现。

    接口：
        encode(*args, **kwargs) -> list[float]

    当前实现只是将输入参数的结构化 repr 映射为哈希向量，
    方便未来替换为真实文本编码模型。
    """

    def encode(self, *args: Any, **kwargs: Any) -> List[float]:
        """将任意文本相关输入编码为定长向量。"""

        payload_tuple: Tuple[Any, ...] = args
        # 为保证稳定性，对 kwargs 做排序
        kwargs_items = tuple(sorted(kwargs.items()))
        payload_str = f"text:{repr((payload_tuple, kwargs_items))}"

        logger.info("文本编码输入: args=%s, kwargs=%s", args, kwargs)
        vector = _hash_to_vector(payload_str)
        logger.info("文本编码结果向量长度=%d", len(vector))
        return vector

