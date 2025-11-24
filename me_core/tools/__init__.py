"""工具（tools）相关模块。

当前提供：
- ToolInfo / ToolRegistry：工具元信息与注册表；
- ToolExecutorStub：用于原型阶段的工具执行桩；
- BaseTool / EchoTool / TimeTool：面向 Agent 的基础工具接口与简单实现；
- MultimodalQATool：多模态问答占位工具，实现“图片+文本问题→文本回答”的统一接口。
"""

from .base import (  # noqa: F401
    BaseTool,
    EchoTool,
    FileReadTool,
    HttpGetTool,
    MultimodalQATool,
    SelfDescribeTool,
    TimeTool,
    ToolSpec,
)
from .registry import ToolInfo, ToolRegistry  # noqa: F401
from .executor_stub import ToolExecutorStub  # noqa: F401
from .codetools import ReadFileTool, WriteFileTool, ApplyPatchTool  # noqa: F401
from .runtools import RunCommandTool, RunTestsTool, RunTrainingScriptTool  # noqa: F401
from .braintools import DumpBrainGraphTool, EvalBrainEnergyTool, EvalBrainMemoryTool  # noqa: F401

__all__ = [
    "BaseTool",
    "ToolSpec",
    "EchoTool",
    "TimeTool",
    "HttpGetTool",
    "FileReadTool",
    "SelfDescribeTool",
    "MultimodalQATool",
    "ReadFileTool",
    "WriteFileTool",
    "ApplyPatchTool",
    "RunCommandTool",
    "RunTestsTool",
    "RunTrainingScriptTool",
    "DumpBrainGraphTool",
    "EvalBrainEnergyTool",
    "EvalBrainMemoryTool",
    "ToolInfo",
    "ToolRegistry",
    "ToolExecutorStub",
]
