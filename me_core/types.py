from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# 通用类型别名：用于在整个项目中复用
JsonDict = Dict[str, Any]


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

    属性：
        timestamp: 事件发生的时间，使用 UTC。
        event_type: 事件类型，推荐使用约定俗成的字符串，例如：
            - "perception"：感知到的新信息
            - "action"：对外执行的行动
            - "internal"：内部推理 / 状态更新
        payload: 事件的具体内容，保持结构化但不过度约束。
    """

    timestamp: datetime
    event_type: str
    payload: Optional[JsonDict] = None

    @staticmethod
    def now(event_type: str, payload: Optional[JsonDict] = None) -> "AgentEvent":
        """使用当前时间快速构造一个事件。

        方便在代码中直接写：
            AgentEvent.now("perception", {"text": "用户输入"})
        """
        return AgentEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            payload=payload,
        )


@dataclass(slots=True)
class ToolCall:
    """一次工具调用请求的基本结构。

    该结构专注于“我想 / 我要”层面：
    - tool_name：我要调用哪个工具；
    - arguments：我打算如何调用它（参数）；
    - call_id：一次调用的唯一标识，用于与结果对齐。

    属性：
        tool_name: 工具名称，通常与工具注册表中的名称一致。
        arguments: 传给工具的参数，使用 JSON 友好的字典。
        call_id: 本次调用的唯一 ID，便于追踪与对齐结果。
        timestamp: 生成调用请求的时间（UTC）。
        meta: 可选元信息，如调用来源、优先级、trace_id 等。
    """

    tool_name: str
    arguments: JsonDict
    call_id: str
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    meta: Optional[JsonDict] = None


@dataclass(slots=True)
class ToolResult:
    """一次工具调用结果的基本结构。

    该结构专注于“我做”的反馈与“我学到什么”的基础材料：
    - success：调用是否成功；
    - output：若成功，工具返回的主要结果；
    - error：若失败，错误信息（面向人类或日志）；
    - meta：额外信息，如耗时、重试次数等。

    属性：
        call_id: 对应的工具调用 ID，需与 ToolCall.call_id 一致。
        success: 是否调用成功。
        output: 工具的输出结果，结构由具体工具自行约定。
        error: 若失败，记录错误消息；若成功则可为 None。
        timestamp: 结果产生的时间（UTC）。
        meta: 可选元信息，如性能指标、调试数据等。
    """

    call_id: str
    success: bool
    output: Optional[JsonDict] = None
    error: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    meta: Optional[JsonDict] = None
