"""顶层智能体（agent）相关模块。

当前提供：
- StateStore：简单的 JSON 状态存储；
- run_once：执行一轮完整的自我学习主循环（旧版流程）；
- BaseAgent / SimpleAgent：面向事件流的最小可运行 Agent 框架。
"""

from .agent_loop import run_once  # noqa: F401
from .simple_agent import BaseAgent, SimpleAgent  # noqa: F401
from .state_store import StateStore  # noqa: F401

__all__ = [
    "run_once",
    "StateStore",
    "BaseAgent",
    "SimpleAgent",
]

