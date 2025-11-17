from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar

# 通用类型别名：用于在整个项目中复用
JsonDict = Dict[str, Any]


class EventSource(str, Enum):
    """事件来源枚举。

    该枚举仅作为“推荐使用”的来源标识，字段取值保持为字符串，便于序列化。
    后续在接入更复杂的智能体编排时，可以扩展更多来源类型：
        - POPULATION: 表示来自“种群智能体”的汇总事件；
        - EXTERNAL_TEACHER: 表示来自外部教师/监督者（例如人类或其他大模型）。
    """

    HUMAN = "human"
    TOOL = "tool"
    AGENT_INTERNAL = "agent_internal"
    ENVIRONMENT = "environment"
    SYSTEM = "system"
    POPULATION = "population"  # TODO: 预留给“种群进化”场景
    EXTERNAL_TEACHER = "external_teacher"  # TODO: 预留给“多教师模式”


class EventKind(str, Enum):
    """事件类型枚举。

    这里给出若干常见类型，避免在不同模块中出现语义不清的魔法字符串。
    """

    PERCEPTION = "perception"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    DIALOGUE = "dialogue"
    DRIVE = "drive"
    LEARNING = "learning"
    TASK = "task"
    INTERNAL = "internal"
    ERROR = "error"


class ToolKind(str, Enum):
    """工具类型枚举。

    当前仅用于标注工具的大致类别，后续可以扩展：
        - EXTERNAL_TEACHER: 外部教师型工具，例如大模型评审器；
        - POPULATION_TOOL: 与种群级别协调相关的工具。
    """

    INTERNAL = "internal"
    EXTERNAL = "external"
    EXTERNAL_TEACHER = "external_teacher"  # TODO: 预留“多教师模式”
    SYSTEM = "system"


TAgentEvent = TypeVar("TAgentEvent", bound="AgentEvent")
TToolCall = TypeVar("TToolCall", bound="ToolCall")
TToolResult = TypeVar("TToolResult", bound="ToolResult")


@dataclass(slots=True)
class MultiModalInput:
    """多模态输入的统一占位结构。

    当前阶段仅存放各模态的元信息与原始文本：
        text: 文本内容（若存在），适合传给文本编码器
        image_meta: 图像的元信息（如文件名、标签等）
        audio_meta: 音频的元信息
        video_meta: 视频的元信息

    后续可以将其扩展为真正的多模态张量或引用。
    """

    text: Optional[str] = None
    image_meta: Optional[JsonDict] = None
    audio_meta: Optional[JsonDict] = None
    video_meta: Optional[JsonDict] = None


@dataclass(slots=True)
class AgentEvent:
    """描述一次智能体的行为或感知事件。

    该结构作为系统内的“事件语言”，用来在各模块之间传递信息。
    典型场景包括：
    - 感知到外部输入（如用户指令、环境变化）
    - 执行动作（如调用工具、发出回复）
    - 内部状态变化（如目标更新、自我反思结果）

    属性（为兼容既有代码，保持字段名称不变并向后扩展）：
        timestamp: 事件发生时间（UTC）。
        event_type: 事件类型字符串，约定与 EventKind 的值保持一致，如：
            - "perception"：感知到的新信息；
            - "tool_call"：发起工具调用；
            - "tool_result"：收到工具结果；
            - "dialogue"：对话输入/输出；
            - "task"：任务级别事件。
        payload: 事件具体内容，保持结构化但不过度约束。
        id: 事件唯一标识，默认使用 UUID4 字符串。
        source: 事件来源（推荐使用 EventSource 的值），例如 "human" / "tool"。
        kind: 事件语义类型（可选），推荐使用 EventKind 的枚举值。
        trace_id: 追踪同一条思考链路的 id，便于未来在“多轮推理/多智能体协作”
                  场景中进行调试与可视化。
        meta: 额外元信息（例如 agent_id、population_id 等），方便未来扩展。
    """

    timestamp: datetime
    event_type: str
    payload: Optional[JsonDict] = None
    id: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: Optional[str] = None
    kind: Optional[EventKind] = None
    trace_id: Optional[str] = None
    meta: JsonDict = field(default_factory=dict)

    @staticmethod
    def now(
        event_type: str,
        payload: Optional[JsonDict] = None,
        *,
        source: Optional[str] = None,
        kind: Optional[EventKind] = None,
        trace_id: Optional[str] = None,
    ) -> "AgentEvent":
        """使用当前时间快速构造一个事件。

        方便在代码中直接写：
            AgentEvent.now("perception", {"text": "用户输入"})
        """
        return AgentEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            payload=payload,
            source=source,
            kind=kind,
            trace_id=trace_id,
        )

    def to_dict(self) -> JsonDict:
        """将事件转换为 JSON 友好的字典。

        说明：
            - datetime 字段统一使用 ISO8601 字符串表示；
            - Enum 字段使用其 value（即底层字符串）；
            - 该方法与 StateStore 等模块共享，用作统一序列化格式。
        """

        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "payload": self.payload,
            "source": self.source,
            "kind": self.kind.value if isinstance(self.kind, EventKind) else self.kind,
            "trace_id": self.trace_id,
            "meta": dict(self.meta) if self.meta is not None else {},
        }

    @classmethod
    def from_dict(cls: Type[TAgentEvent], data: JsonDict) -> TAgentEvent:
        """从字典结构恢复 AgentEvent。

        为了兼容旧数据：
            - 若时间解析失败或缺失，则使用“当前时间”兜底；
            - 若缺失 event_type，则使用 "unknown"；
            - 对不存在的字段使用合理默认值。
        """

        ts_raw = data.get("timestamp")
        if isinstance(ts_raw, str):
            try:
                timestamp = datetime.fromisoformat(ts_raw)
            except ValueError:
                timestamp = datetime.now(timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        event_type = str(data.get("event_type") or "unknown")
        payload = data.get("payload")

        kind_raw = data.get("kind")
        kind: Optional[EventKind]
        if isinstance(kind_raw, str):
            try:
                kind = EventKind(kind_raw)
            except ValueError:
                kind = None
        else:
            kind = None

        return cls(
            timestamp=timestamp,
            event_type=event_type,
            payload=payload,
            id=str(data.get("id") or datetime.now(timezone.utc).isoformat()),
            source=data.get("source"),
            kind=kind,
            trace_id=data.get("trace_id"),
            meta=dict(data.get("meta") or {}),
        )

    def pretty(self) -> str:
        """以中文格式化当前事件，便于日志打印与调试。"""

        ts = self.timestamp.isoformat()
        src = self.source or "-"
        kind = self.kind.value if isinstance(self.kind, EventKind) else (self.kind or self.event_type)
        trace = self.trace_id or "-"
        payload_keys = ", ".join(sorted(self.payload.keys())) if isinstance(self.payload, dict) else "-"
        return f"[事件] 时间={ts} 来源={src} 类型={kind} trace_id={trace} 载荷键={payload_keys}"

    def __str__(self) -> str:  # pragma: no cover - 简单代理 pretty
        return self.pretty()


@dataclass(slots=True)
class ToolCall:
    """一次工具调用请求的基本结构。

    该结构专注于“我想 / 我要”层面：
    - tool_name：我要调用哪个工具；
    - arguments：我打算如何调用它（参数）；
    - call_id：一次调用的唯一标识，用于与结果对齐。

    属性（为兼容既有实现，这里沿用原有字段命名）：
        tool_name: 工具名称，通常与工具注册表中的名称一致。
        arguments: 传给工具的参数，使用 JSON 友好的字典。
        call_id: 本次调用的唯一 ID，便于追踪与对齐结果。
        timestamp: 生成调用请求的时间（UTC）。
        meta: 可选元信息，如调用来源、优先级、trace_id 等。
        kind: 工具类型标签（可选），推荐使用 ToolKind。

    同时提供若干便捷属性：
        - id / name / args / created_at：分别对应 call_id / tool_name / arguments / timestamp，
          方便在更通用的上下文中使用。
    """

    tool_name: str
    arguments: JsonDict
    call_id: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    meta: JsonDict = field(default_factory=dict)
    kind: Optional[ToolKind] = None

    @property
    def id(self) -> str:
        """工具调用的统一 ID 别名。"""

        return self.call_id

    @property
    def name(self) -> str:
        """工具名称的别名，便于与 BaseTool 等接口统一。"""

        return self.tool_name

    @property
    def args(self) -> JsonDict:
        """调用参数的别名。"""

        return self.arguments

    @property
    def created_at(self) -> datetime:
        """调用创建时间的别名。"""

        return self.timestamp

    def to_dict(self) -> JsonDict:
        """将工具调用结构化为字典，便于持久化或日志记录。"""

        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "call_id": self.call_id,
            "timestamp": self.timestamp.isoformat(),
            "meta": dict(self.meta) if self.meta is not None else {},
            "kind": self.kind.value if isinstance(self.kind, ToolKind) else self.kind,
        }

    @classmethod
    def from_dict(cls: Type[TToolCall], data: JsonDict) -> TToolCall:
        """从字典恢复 ToolCall。

        兼容多种字段命名风格：
            - tool_name / name
            - arguments / args
            - call_id / id
            - timestamp / created_at
        """

        name = data.get("tool_name") or data.get("name") or ""
        args = data.get("arguments") or data.get("args") or {}
        call_id = data.get("call_id") or data.get("id") or ""

        ts_raw = data.get("timestamp") or data.get("created_at")
        if isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        kind_raw = data.get("kind")
        kind: Optional[ToolKind]
        if isinstance(kind_raw, str):
            try:
                kind = ToolKind(kind_raw)
            except ValueError:
                kind = None
        else:
            kind = None

        return cls(
            tool_name=str(name),
            arguments=dict(args),
            call_id=str(call_id),
            timestamp=ts,
            meta=dict(data.get("meta") or {}),
            kind=kind,
        )

    def pretty(self) -> str:
        """以中文格式化工具调用，便于日志打印。"""

        ts = self.timestamp.isoformat()
        kind = self.kind.value if isinstance(self.kind, ToolKind) else (self.kind or "-")
        arg_keys = ", ".join(sorted(self.arguments.keys()))
        return f"[工具调用] 时间={ts} 工具={self.tool_name} 类型={kind} 参数键={arg_keys}"

    def __str__(self) -> str:  # pragma: no cover - 简单代理 pretty
        return self.pretty()


@dataclass(slots=True)
class ToolResult:
    """一次工具调用结果的基本结构。

    该结构专注于“我做”的反馈与“我学到什么”的基础材料：
    - success：调用是否成功；
    - output：若成功，工具返回的主要结果；
    - error：若失败，错误信息（面向人类或日志）；
    - meta：额外信息，如耗时、重试次数等。

    属性（为兼容既有实现，这里沿用原有字段命名）：
        call_id: 对应的工具调用 ID，需与 ToolCall.call_id 一致。
        success: 是否调用成功。
        output: 工具的输出结果，结构由具体工具自行约定。
        error: 若失败，记录错误消息；若成功则可为 None。
        timestamp: 结果产生的时间（UTC）。
        meta: 可选元信息，如性能指标、调试数据等。
    """

    call_id: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    meta: JsonDict = field(default_factory=dict)

    @property
    def finished_at(self) -> datetime:
        """结果产生时间的别名，便于与“任务生命周期”类术语统一。"""

        return self.timestamp

    def to_dict(self) -> JsonDict:
        """将工具结果转换为 JSON 友好的字典结构。"""

        return {
            "call_id": self.call_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "meta": dict(self.meta) if self.meta is not None else {},
        }

    @classmethod
    def from_dict(cls: Type[TToolResult], data: JsonDict) -> TToolResult:
        """从字典恢复 ToolResult。

        若缺失时间字段，则使用当前时间。
        """

        ts_raw = data.get("timestamp") or data.get("finished_at")
        if isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw)
            except ValueError:
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        return cls(
            call_id=str(data.get("call_id") or data.get("id") or ""),
            success=bool(data.get("success")),
            output=data.get("output"),
            error=data.get("error"),
            timestamp=ts,
            meta=dict(data.get("meta") or {}),
        )

    def pretty(self) -> str:
        """以中文格式化工具调用结果，便于日志打印。"""

        ts = self.timestamp.isoformat()
        status = "成功" if self.success else "失败"
        err = self.error or "-"
        out_type = type(self.output).__name__ if self.output is not None else "None"
        return f"[工具结果] 时间={ts} 调用ID={self.call_id} 状态={status} 输出类型={out_type} 错误={err}"

    def __str__(self) -> str:  # pragma: no cover - 简单代理 pretty
        return self.pretty()
