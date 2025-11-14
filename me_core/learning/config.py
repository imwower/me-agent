from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LearningConfig:
    """学习模块相关的可调参数配置。"""

    # 学习意愿阈值：低于该值则不主动调用学习工具
    desire_threshold: float = 0.2

    # 知识库最多保留多少条记录
    max_knowledge_entries: int = 200


DEFAULT_LEARNING_CONFIG = LearningConfig()

