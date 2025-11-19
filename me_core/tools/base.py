from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict

from me_core.types import ToolCall, ToolResult, ImageRef


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


@dataclass(slots=True)
class MultimodalQATool(BaseTool):
    """多模态问答工具桩实现。

    设计意图：
        - 作为“图像 + 文本问题 -> 文本回答”的统一封装；
        - 当前版本不调用真实多模态模型，仅回显简单描述；
        - 未来可在独立扩展模块中接入 CLIP/多模态 LLM 等，再通过同名接口挂载。
    """

    name: str = "multimodal_qa"

    def call(self, call: ToolCall) -> ToolResult:
        image_path = None
        question = ""

        args = call.args
        img_raw = args.get("image")
        if isinstance(img_raw, dict) and img_raw.get("path"):
            image_path = str(img_raw.get("path"))
        elif isinstance(img_raw, str):
            image_path = img_raw

        if isinstance(args.get("question"), str):
            question = args["question"]

        # 当前仅生成占位式回答，说明架构路线
        if image_path and question:
            answer = (
                f"这是一个多模态问答占位工具。我收到的问题是「{question}」，"
                f"并知道你提供了一张图片（路径：{image_path}）。"
                "在未来接入真实多模态模型后，我会尝试基于图片内容和问题生成更准确的回答。"
            )
        elif question:
            answer = (
                f"多模态问答工具目前只收到文本问题「{question}」，"
                "但没有图片信息。未来可以同时提供图片路径与问题，让我尝试做基于图片的回答。"
            )
        else:
            answer = "多模态问答工具当前未收到有效的图片或问题输入。"

        output: Dict[str, object] = {
            "tool_name": self.name,
            "answer": answer,
            "image_path": image_path,
            "question": question,
        }
        return ToolResult(
            call_id=call.call_id,
            success=True,
            output=output,
            error=None,
            meta={"kind": "multimodal_qa"},
        )
