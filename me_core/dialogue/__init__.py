"""对话（dialogue）相关模块。

当前仅提供一个简单的对话规划器与生成桩，用于：
- 根据驱动力与自我总结，决定是否主动发起对话；
- 生成一段中文自述或求助信息。
"""

from .planner import DialoguePlanner, InitiativeDecision  # noqa: F401
from .generator_stub import generate_message  # noqa: F401

__all__ = [
    "DialoguePlanner",
    "InitiativeDecision",
    "generate_message",
]

