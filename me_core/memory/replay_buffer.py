from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

Transition = Tuple[Any, Any, Any, float, bool]


@dataclass
class ReplayBuffer:
    """用于存储环境转移的简单回放缓冲区。

    每条转移的结构为：
        (obs, action, next_obs, reward, done)

    当前实现仅支持：
        - add_transition: 追加单条转移；
        - sample: 随机采样若干条转移，用于 world_model 或策略学习。
    """

    capacity: int = 10000
    _buffer: List[Transition] = field(default_factory=list, init=False)

    def add_transition(
        self,
        obs: Any,
        action: Any,
        next_obs: Any,
        reward: float,
        done: bool,
    ) -> None:
        """向缓冲区追加一条转移。"""

        if len(self._buffer) >= self.capacity:
            # 简单的环形缓冲实现：丢弃最旧的一条
            removed = self._buffer.pop(0)
            logger.info("ReplayBuffer: 达到容量上限，丢弃最旧转移: %s", removed)

        self._buffer.append((obs, action, next_obs, reward, done))
        logger.info("ReplayBuffer: 当前转移数量=%d", len(self._buffer))

    def size(self) -> int:
        """返回当前缓冲区中的转移数量。"""

        return len(self._buffer)

    def sample(self, batch_size: int) -> List[Transition]:
        """随机采样若干条转移。

        若缓冲区内转移数量少于 batch_size，则返回全部转移。
        """

        if batch_size <= 0 or not self._buffer:
            return []

        if batch_size >= len(self._buffer):
            return list(self._buffer)

        samples = random.sample(self._buffer, batch_size)
        logger.info("ReplayBuffer: 采样 batch_size=%d, 实际返回=%d", batch_size, len(samples))
        return samples

