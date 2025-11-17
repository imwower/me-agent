"""对话（dialogue）相关模块。

当前提供：
- DialoguePlanner / generate_message：基于驱动力的主动发言规划与生成桩；
- BaseDialoguePolicy / RuleBasedDialoguePolicy：面向 Agent 的对话策略接口与实现。
"""

from .planner import DialoguePlanner, InitiativeDecision  # noqa: F401
from .generator_stub import generate_message  # noqa: F401
from .policy import BaseDialoguePolicy, RuleBasedDialoguePolicy  # noqa: F401

__all__ = [
    "DialoguePlanner",
    "InitiativeDecision",
    "generate_message",
    "BaseDialoguePolicy",
    "RuleBasedDialoguePolicy",
]

