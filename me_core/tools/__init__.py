"""工具（tools）相关模块。

当前提供：
- ToolInfo / ToolRegistry：工具元信息与注册表；
- ToolExecutorStub：用于原型阶段的工具执行桩；
- BaseTool / EchoTool / TimeTool：面向 Agent 的基础工具接口与简单实现。
"""

from .base import BaseTool, EchoTool, TimeTool  # noqa: F401
from .registry import ToolInfo, ToolRegistry  # noqa: F401
from .executor_stub import ToolExecutorStub  # noqa: F401

__all__ = [
    "BaseTool",
    "EchoTool",
    "TimeTool",
    "ToolInfo",
    "ToolRegistry",
    "ToolExecutorStub",
]
