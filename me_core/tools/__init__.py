"""工具（tools）相关模块。

当前提供：
- ToolInfo / ToolRegistry：工具元信息与注册表。
- ToolExecutorStub：用于原型阶段的工具执行桩。
"""

from .registry import ToolInfo, ToolRegistry  # noqa: F401
from .executor_stub import ToolExecutorStub  # noqa: F401

__all__ = [
    "ToolInfo",
    "ToolRegistry",
    "ToolExecutorStub",
]

