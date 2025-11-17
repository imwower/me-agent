from __future__ import annotations

import hashlib
import logging
from typing import Any, List

logger = logging.getLogger(__name__)


def _to_bytes(value: Any) -> bytes:
    """将任意可序列化对象转换为字节串。

    当前实现使用 repr(value).encode("utf-8")，保证在不同进程中稳定，
    避免 Python 内置 hash 的随机种子影响。
    """

    return repr(value).encode("utf-8")


def encode_obs(obs: Any, dim: int = 16) -> List[float]:
    """将环境观测编码为定长向量。

    设计目标：
        - 使用完全基于标准库的方式构造一个稳定的 embedding；
        - 相同 obs 在不同运行中得到相同向量；
        - 不追求语义意义，只作为 world_model 自监督的输入空间。

    实现思路：
        - 使用 hashlib.sha256 对 obs 的字节表示求哈希；
        - 将哈希结果切分为 dim 段，每段映射到 [0,1] 之间的浮点数；
        - 返回长度为 dim 的浮点列表。
    """

    raw = _to_bytes(obs)
    digest = hashlib.sha256(raw).digest()

    # 每个字节映射到 [0,1]，然后按需截断或重复到指定维度
    vals = [b / 255.0 for b in digest]
    if len(vals) >= dim:
        vec = vals[:dim]
    else:
        # 若 digest 不足以覆盖 dim（SHA256 实际上长度足够），则循环填充
        vec = (vals * (dim // len(vals) + 1))[:dim]

    logger.info("对环境观测进行编码: obs=%r, dim=%d", obs, dim)
    return vec


def encode_transition(
    obs: Any,
    action: Any,
    next_obs: Any,
    dim: int = 16,
) -> tuple[list[float], list[float]]:
    """辅助函数：同时编码当前观测与下一步观测。

    返回：
        (obs_embed, next_obs_embed)
    """

    return encode_obs(obs, dim=dim), encode_obs(next_obs, dim=dim)

