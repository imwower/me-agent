from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict

from me_core.types import ToolCall, ToolResult


class BaseTool(ABC):
    """工具接口基类。

    设计意图：
        - 将具体外部能力（如 echo、时间查询等）封装为统一接口；
        - 输入使用 ToolCall，输出使用 ToolResult，方便纳入事件流；
        - 未来可以扩展为异步/远程调用，而不影响 Agent 侧逻辑。
    """

    name: str

    @abstractmethod
    def call(self, call: ToolCall) -> ToolResult:
        """执行一次工具调用并返回结构化结果。"""


@dataclass(slots=True)
class EchoTool(BaseTool):
    """简单回声工具。

    功能：
        - 读取调用参数中的 "text" 字段；
        - 将其原样写回到 ToolResult.output 中；
        - 主要用于验证 ToolCall / ToolResult 流程是否连通。
    """

    name: str = "echo"

    def call(self, call: ToolCall) -> ToolResult:
        text = ""
        if isinstance(call.args.get("text"), str):
            text = call.args["text"]
        else:
            # 若调用方未显式提供 text，则将全部参数以字符串形式回显
            text = str(call.args)

        output: Dict[str, object] = {
            "tool_name": self.name,
            "text": text,
        }
        return ToolResult(
            call_id=call.call_id,
            success=True,
            output=output,
            error=None,
            meta={"kind": "echo"},
        )


@dataclass(slots=True)
class TimeTool(BaseTool):
    """当前时间查询工具。

    功能：
        - 返回当前 UTC 时间的 ISO 字符串；
        - 便于在 demo 中展示“调用工具获取环境信息”的最小例子。
    """

    name: str = "time"

    def call(self, call: ToolCall) -> ToolResult:
        _ = call  # 当前实现未使用调用参数，仅预留扩展位
        now = datetime.now(timezone.utc)
        output: Dict[str, object] = {
            "tool_name": self.name,
            "now_iso": now.isoformat(),
        }
        return ToolResult(
            call_id=call.call_id,
            success=True,
            output=output,
            error=None,
            meta={"kind": "time"},
        )

