"""me-agent 核心包。

当前仅提供基础数据结构定义，用于描述智能体的事件与工具调用。
后续会在此包内逐步添加各功能模块的公共接口，例如：

- 感知（perception）
- 世界模型（world_model）
- 自我模型（self_model）
- 内在驱动力（drives）
- 工具调用（tools）
- 学习与记忆（learning）
- 对话（dialogue）
- 智能体编排（agent）
"""

from .types import AgentEvent, ToolCall, ToolResult

__all__ = [
    "AgentEvent",
    "ToolCall",
    "ToolResult",
]

