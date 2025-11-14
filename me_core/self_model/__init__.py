"""自我模型（self_model）相关模块。

当前提供：
- SelfState：描述智能体对“自我”的当前认识。
- SelfUpdater：根据事件更新自我状态。
- SelfSummarizer：从自我状态生成简要自述。
"""

from .self_state import SelfState  # noqa: F401
from .self_summarizer import summarize_self  # noqa: F401
from .self_updater import aggregate_stats, update_from_event  # noqa: F401

__all__ = [
    "SelfState",
    "summarize_self",
    "update_from_event",
    "aggregate_stats",
]

