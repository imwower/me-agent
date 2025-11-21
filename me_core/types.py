from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar, NewType, Set, Literal, List

# 为了在类型层集中暴露核心状态结构，这里仅定义/聚合“轻量数据结构”。
# 复杂的更新逻辑仍然放在各自子模块中（如 self_model / drives 等），
# 以避免 types.py 演化成一个“上帝模块”。

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

# SelfState / DriveVector 的完整定义位于各自子包中，
# 在此仅做类型级引用，作为 AgentState 等聚合结构的组成部分。
try:  # 避免在类型定义早期导入时产生循环依赖
    from me_core.self_model.self_state import SelfState  # type: ignore  # noqa: WPS433
    from me_core.drives.drive_vector import DriveVector  # type: ignore  # noqa: WPS433
except Exception:  # pragma: no cover - 仅在极早期导入/静态分析时触发
    SelfState = object  # type: ignore
    DriveVector = object  # type: ignore


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


# 轻量级多模态引用结构 ----------------------------------------------------------------

ImageId = NewType("ImageId", str)
AudioId = NewType("AudioId", str)


@dataclass(slots=True)
class ImageRef:
    """图像引用的统一结构。

    说明：
        - path: 本地路径或 URL（由上层约定），不强制要求一定可直接打开；
        - width/height: 若已知则填入，否则可为 None；
        - meta: 其他元信息，例如来源、文件大小、hash 等。
    """

    path: str
    width: Optional[int] = None
    height: Optional[int] = None
    meta: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class AudioRef:
    """音频引用的统一结构。

    当前仅记录基本的文件路径与时长/采样率信息，具体波形或特征由上层模块负责。
    """

    path: str
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    meta: JsonDict = field(default_factory=dict)


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

    为支持多模态对齐与概念空间，新增字段（保持向后兼容）：
        modality: 当前事件主模态，例如 "text" / "image" / "audio" / "mixed"；
        embedding: 可选的向量表示（通常由对齐模块填充），用于在概念空间中检索；
        tags: 任意标签集合，便于基于标签的检索与学习。
    """

    timestamp: datetime
    event_type: str
    payload: JsonDict = field(default_factory=dict)
    id: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: Literal["user", "environment", "tool"] | str = "user"
    kind: Optional[EventKind] = None
    trace_id: Optional[str] = None
    meta: JsonDict = field(default_factory=dict)
    modality: Literal["text", "image", "audio", "mixed"] | str = "text"
    embedding: Optional[List[float]] = None
    tags: Set[str] = field(default_factory=set)

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
        payload_data = payload if isinstance(payload, dict) else {}
        return AgentEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            payload=payload_data,
            source=source or "user",
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
            "payload": dict(self.payload) if self.payload is not None else {},
            "source": self.source,
            "kind": self.kind.value if isinstance(self.kind, EventKind) else self.kind,
            "trace_id": self.trace_id,
            "meta": dict(self.meta) if self.meta is not None else {},
            "modality": self.modality,
            "embedding": list(self.embedding) if isinstance(self.embedding, list) else self.embedding,
            "tags": sorted(self.tags),
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
        payload_raw = data.get("payload")
        payload: JsonDict = dict(payload_raw) if isinstance(payload_raw, dict) else {}

        kind_raw = data.get("kind")
        kind: Optional[EventKind]
        if isinstance(kind_raw, str):
            try:
                kind = EventKind(kind_raw)
            except ValueError:
                kind = None
        else:
            kind = None

        tags_raw = data.get("tags") or []
        if isinstance(tags_raw, (list, set, tuple)):
            tags_set: Set[str] = {str(t) for t in tags_raw}
        else:
            tags_set = set()

        embedding_raw = data.get("embedding")
        embedding: Optional[List[float]]
        if isinstance(embedding_raw, list):
            embedding = [float(x) for x in embedding_raw]
        else:
            embedding = None

        modality = data.get("modality") or "text"
        source_val = data.get("source") or "user"

        return cls(
            timestamp=timestamp,
            event_type=event_type,
            payload=payload,
            id=str(data.get("id") or datetime.now(timezone.utc).isoformat()),
            source=source_val,
            kind=kind,
            trace_id=data.get("trace_id"),
            meta=dict(data.get("meta") or {}),
            modality=modality,
            embedding=embedding,
            tags=tags_set,
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


@dataclass(slots=True)
class ToolStats:
    """工具使用统计信息。

    字段：
        usage_count: 调用总次数；
        success_count: 成功次数；
        avg_gain: 平均收益（例如预测误差降低量的平均值）；
        last_used_step: 最近一次被调用时的全局 step 编号。

    该结构将作为 ToolProgram 内部的“表现记录”，供学习与进化模块使用。
    """

    usage_count: int = 0
    success_count: int = 0
    avg_gain: float = 0.0
    last_used_step: int = -1


@dataclass(slots=True)
class ToolProgram:
    """表示一个可被调用的“工具程序”。

    字段：
        name: 工具名称（在 ToolLibrary 中应唯一）；
        dsl_source: 工具内部 DSL 源码或序列表示；
        input_spec: 输入规范（例如参数名称列表），当前用字典占位；
        output_spec: 输出规范（例如返回字段说明），当前用字典占位；
        parents: 产生该工具的“父工具”名称列表，用于追踪进化谱系；
        stats: 与该工具相关的统计信息（ToolStats）。

    说明：
        - 在早期阶段，dsl_source 可以是简单的 JSON 序列或伪代码字符串；
        - 后续 Phase 会在 tools.dsl 模块中引入真正的 DSL 定义与解析逻辑。
    """

    name: str
    dsl_source: str
    input_spec: JsonDict = field(default_factory=dict)
    output_spec: JsonDict = field(default_factory=dict)
    parents: list[str] = field(default_factory=list)
    stats: ToolStats = field(default_factory=ToolStats)


@dataclass(slots=True)
class AgentState:
    """智能体内部的聚合状态结构。

    字段：
        self_state: 自我模型状态（SelfState）；
        drives: 内在驱动力向量（DriveVector）；
        world_model_state: 世界模型内部状态的轻量表示（例如参数摘要、统计信息）；
        memory_state: 记忆系统状态（例如最近轨迹摘要、事件计数等）；
        tool_library_state: 工具库状态（例如现有 ToolProgram 名单及其统计）；
        global_step: 个体内部全局步数计数器，用于对齐各模块中的“最近使用步数”等信息；
        env_state_summary: 环境对该个体暴露的高层状态摘要（例如关卡 ID、场景标识）。

    注意：
        - world_model_state / memory_state / tool_library_state / env_state_summary
          在当前阶段仅作为“信息容器”，具体字段由各子模块自行约定；
        - 这里统一使用 JsonDict，以保持序列化简单、接口稳定。
    """

    self_state: "SelfState"
    drives: "DriveVector"
    world_model_state: JsonDict = field(default_factory=dict)
    memory_state: JsonDict = field(default_factory=dict)
    tool_library_state: JsonDict = field(default_factory=dict)
    global_step: int = 0
    env_state_summary: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class Genotype:
    """用于编码一个个体“先天配置”的基因型结构。

    字段：
        id: 基因型自身的标识（可选），便于追踪谱系；
        parent_ids: 产生该基因型的父基因型 ID 列表；
        world_model_config: 世界模型相关配置（结构自由）；
        learning_config: 学习与元学习相关配置；
        drive_baseline: 驱动力基线（例如 DriveVector 的默认值字典形式）；
        tool_config: 工具系统相关配置（例如初始工具集、DSL 超参数等）。

    Genotype 在种群层面用于：
        - 初始化个体的 AgentCore / AgentState；
        - 在进化阶段进行变异/交叉，生成新的候选个体。
    """

    id: Optional[str] = None
    parent_ids: list[str] = field(default_factory=list)
    world_model_config: JsonDict = field(default_factory=dict)
    learning_config: JsonDict = field(default_factory=dict)
    drive_baseline: JsonDict = field(default_factory=dict)
    tool_config: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class Individual:
    """种群中的单个个体。

    字段：
        id: 个体唯一标识；
        agent_state: 当前智能体内部状态聚合（AgentState）；
        genotype: 该个体的基因型配置（Genotype）；
        fitness: 当前估计的适应度值；
        age: 已经历的生命周期步数（以环境 step 为单位）；
        generation: 所处进化代数（由 PopulationManager 维护）；
        parent_ids: 产生该个体的父个体 ID 列表；
        env_id: 该个体主要评估环境的标识（可选）；
        eval_count: 对该个体进行评估的次数；
        frozen: 若为 True，表示该个体在后续进化中不再参与变异，只作为参考/基线。

    PopulationManager 将围绕 Individual 进行评估、选择、变异和繁衍。
    """

    id: str
    agent_state: AgentState
    genotype: Genotype
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    env_id: Optional[str] = None
    eval_count: int = 0
    frozen: bool = False
