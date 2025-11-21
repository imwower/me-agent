from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable
from urllib import parse, request

from me_core.types import ToolCall, ToolResult


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


@runtime_checkable
class BaseTool(Protocol):
    """工具接口协议。"""

    spec: ToolSpec

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @property
    def name(self) -> str:
        ...

    def call(self, call: ToolCall) -> ToolResult:
        """兼容旧接口的包装。"""
        ...


@dataclass(slots=True)
class EchoTool:
    """简单回声工具。"""

    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="echo",
            description="回显输入文本，主要用于测试",
            input_schema={"text": "string"},
            output_schema={"echo": "string"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        text = params.get("text")
        if not isinstance(text, str):
            text = str(params)
        return {"tool_name": self.name, "echo": text}

    def call(self, call: ToolCall) -> ToolResult:
        output = self.run(call.arguments)
        return ToolResult(
            call_id=call.call_id,
            success=True,
            output=output,
            error=None,
            meta={"kind": "echo"},
        )


@dataclass(slots=True)
class TimeTool:
    """当前时间查询工具。"""

    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="time",
            description="返回当前 UTC 时间",
            input_schema={},
            output_schema={"now_iso": "string"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _ = params
        now = datetime.now(timezone.utc)
        return {"tool_name": self.name, "now_iso": now.isoformat()}

    def call(self, call: ToolCall) -> ToolResult:
        output = self.run(call.arguments)
        return ToolResult(
            call_id=call.call_id,
            success=True,
            output=output,
            error=None,
            meta={"kind": "time"},
        )


@dataclass(slots=True)
class MultimodalQATool:
    """多模态问答工具桩实现。"""

    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="multimodal_qa",
            description="占位式多模态问答工具",
            input_schema={"image": "dict|path", "question": "string"},
            output_schema={"answer": "string"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image_path = None
        question = ""

        img_raw = params.get("image")
        if isinstance(img_raw, dict) and img_raw.get("path"):
            image_path = str(img_raw.get("path"))
        elif isinstance(img_raw, str):
            image_path = img_raw

        if isinstance(params.get("question"), str):
            question = params["question"]

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

        return {
            "tool_name": self.name,
            "answer": answer,
            "image_path": image_path,
            "question": question,
        }

    def call(self, call: ToolCall) -> ToolResult:
        output = self.run(call.arguments)
        return ToolResult(
            call_id=call.call_id,
            success=True,
            output=output,
            error=None,
            meta={"kind": "multimodal_qa"},
        )


@dataclass(slots=True)
class HttpGetTool:
    """使用标准库执行简化版 HTTP GET 的工具。"""

    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="http_get",
            description="使用 urllib.request 发送 GET 请求（简化版）",
            input_schema={"url": "string", "params": "dict(optional)"},
            output_schema={"status_code": "int", "headers": "dict", "body": "string"},
        )
    )
    timeout: float = 5.0
    max_body_length: int = 800

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        url = params.get("url")
        query = params.get("params")
        if not isinstance(url, str):
            raise ValueError("url 参数必须是字符串")
        final_url = url
        if isinstance(query, dict) and query:
            qs = parse.urlencode({k: str(v) for k, v in query.items()})
            delimiter = "&" if "?" in url else "?"
            final_url = f"{url}{delimiter}{qs}"

        req = request.Request(final_url, method="GET")
        with request.urlopen(req, timeout=self.timeout) as resp:
            body_bytes = resp.read(self.max_body_length + 1)
            body = body_bytes[: self.max_body_length].decode("utf-8", errors="ignore")
            headers = dict(resp.headers.items())
            return {
                "tool_name": self.name,
                "status_code": resp.getcode(),
                "headers": headers,
                "body": body,
            }

    def call(self, call: ToolCall) -> ToolResult:
        try:
            output = self.run(call.arguments)
            success = True
            error = None
        except Exception as exc:  # pragma: no cover - 网络环境相关
            output = {"tool_name": self.name}
            success = False
            error = str(exc)
        return ToolResult(
            call_id=call.call_id,
            success=success,
            output=output,
            error=error,
            meta={"kind": "http_get"},
        )


@dataclass(slots=True)
class FileReadTool:
    """读取本地文本文件的工具。"""

    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="file_read",
            description="读取本地文件内容（纯文本，长度受限）",
            input_schema={"path": "string"},
            output_schema={"content": "string"},
        )
    )
    max_length: int = 2000

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        path_raw = params.get("path")
        if not isinstance(path_raw, str):
            raise ValueError("path 参数必须是字符串")
        path = Path(path_raw)
        content = path.read_text(encoding="utf-8")
        truncated = content[: self.max_length]
        return {"tool_name": self.name, "content": truncated}

    def call(self, call: ToolCall) -> ToolResult:
        try:
            output = self.run(call.arguments)
            success = True
            error = None
        except Exception as exc:
            output = {"tool_name": self.name}
            success = False
            error = str(exc)
        return ToolResult(
            call_id=call.call_id,
            success=success,
            output=output,
            error=error,
            meta={"kind": "file_read"},
        )


@dataclass(slots=True)
class SelfDescribeTool:
    """调用自我描述接口的工具。"""

    self_model: Any
    world_model: Any | None = None
    spec: ToolSpec = field(
        default_factory=lambda: ToolSpec(
            name="self_describe",
            description="返回当前智能体的自述",
            input_schema={},
            output_schema={"description": "string"},
        )
    )

    @property
    def name(self) -> str:
        return self.spec.name

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        _ = params
        describer = getattr(self.self_model, "describe_self", None)
        if callable(describer):
            text = describer(self.world_model)
        else:
            text = str(self.self_model)
        return {"tool_name": self.name, "description": text}

    def call(self, call: ToolCall) -> ToolResult:
        output = self.run(call.arguments)
        return ToolResult(
            call_id=call.call_id,
            success=True,
            output=output,
            error=None,
            meta={"kind": "self_describe"},
        )
