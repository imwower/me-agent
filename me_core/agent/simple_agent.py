from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from me_core.event_stream import EventStream
from me_core.types import AgentEvent, EventKind, ToolCall, ToolResult

from me_core.alignment.concepts import ConceptSpace
from me_core.alignment.embeddings import DummyEmbeddingBackend
from me_core.alignment.aligner import MultimodalAligner

from ..dialogue import BaseDialoguePolicy
from ..drives import BaseDriveSystem, Intent
from ..learning import BaseLearner
from ..perception import ImagePerception
from ..perception.base import BasePerception
from ..self_model import BaseSelfModel
from ..tools import BaseTool
from ..world_model import BaseWorldModel


class BaseAgent:
    """Agent 抽象基类。

    对外暴露统一的交互接口：
        - step(raw_input): 执行一次“感知 → 决策 → 行动 → 对话”的最小闭环；
        - run(): 可选的简单命令行循环，便于快速演示。
    """

    def step(self, raw_input: Optional[Any], image_path: Optional[str] = None) -> Optional[str]:  # pragma: no cover - 接口定义
        """处理一次外部输入，返回对话回复或 None。"""

        raise NotImplementedError

    def run(self) -> None:  # pragma: no cover - 交互循环主要依赖 demo 脚本
        """默认实现的简单命令行循环。

        实际 demo 建议使用 scripts/demo_cli_agent.py 中的封装入口。
        """

        print("=== me-agent 简易交互循环（BaseAgent） ===")  # noqa: T201
        while True:
            try:
                user_text = input("你: ").strip()
            except EOFError:
                break
            if user_text.lower() in {"exit", "quit", "q"}:
                print("再见～")  # noqa: T201
                break
            if not user_text:
                continue
            reply = self.step(user_text)
            if reply:
                print(f"Agent: {reply}")  # noqa: T201


@dataclass
class SimpleAgent(BaseAgent):
    """轻量但完整的 Agent 实现。

    该实现串联起：
        - 感知模块（perception）
        - 世界模型（world_model）
        - 自我模型（self_model）
        - 驱动力系统（drives）
        - 工具层（tools）
        - 学习模块（learning）
        - 对话策略（dialogue）
        - 事件流（event_stream）

    目标是在最小复杂度下，展示一个“我想 / 我要 / 我做”的完整回合。
    """

    perception: BasePerception
    world_model: BaseWorldModel
    self_model: BaseSelfModel
    drive_system: BaseDriveSystem
    tools: Dict[str, BaseTool]
    learner: BaseLearner
    dialogue_policy: BaseDialoguePolicy
    event_stream: EventStream = field(default_factory=EventStream)
    image_perception: Optional[BasePerception] = field(default=None, init=False, repr=False)
    concept_space: ConceptSpace = field(default_factory=ConceptSpace, init=False, repr=False)
    embedding_backend: DummyEmbeddingBackend = field(
        default_factory=lambda: DummyEmbeddingBackend(dim=64), init=False, repr=False
    )
    # 为后续“种群进化 / 多教师模式”等扩展预留标识字段
    agent_id: str = "default"
    population_id: Optional[str] = None

    # 便于调试与测试的最后一轮内部状态记录
    last_intent: Optional[Intent] = field(default=None, init=False)
    last_tool_result: Optional[ToolResult] = field(default=None, init=False)

    # 多模态对齐器（默认使用 DummyEmbeddingBackend + 内部 ConceptSpace）
    aligner: Optional[MultimodalAligner] = field(default=None, init=False)

    def __post_init__(self) -> None:
        # 设定默认的多模态对齐组件与图谱
        if self.aligner is None:
            self.aligner = MultimodalAligner(
                backend=self.embedding_backend,
                concept_space=self.concept_space,
                similarity_threshold=0.8,
            )
        else:
            self.concept_space = self.aligner.concept_space
            backend = getattr(self.aligner, "backend", None)
            if backend is not None:
                try:
                    self.embedding_backend = backend  # type: ignore[assignment]
                except Exception:
                    pass

        # world_model 持有同一个 concept_space，避免重复
        if hasattr(self.world_model, "concept_space"):
            self.world_model.concept_space = self.concept_space  # type: ignore[attr-defined]

        # 默认准备一个基础的图片感知器，便于 demo 直接使用
        if self.image_perception is None:
            try:
                self.image_perception = ImagePerception()
            except Exception:
                self.image_perception = None

        # 预先标记 Dummy 多模态对齐能力标签
        try:
            self.self_model.get_state().capability_tags.add("dummy_multimodal_alignment")
        except Exception:
            pass

    def _register_event(self, event: AgentEvent, concept: Optional[Any] = None) -> None:
        """集中处理事件写入、世界/自我模型更新与日志。"""

        event.meta.setdefault("agent_id", self.agent_id)
        if self.population_id is not None:
            event.meta.setdefault("population_id", self.population_id)

        self.event_stream.append_event(event)
        self.event_stream.log_event(event)

        observer = getattr(self.world_model, "observe_event", None)
        if callable(observer):
            observer(event, concept)

        if hasattr(self.self_model, "observe_event"):
            self.self_model.observe_event(event)  # type: ignore[attr-defined]
        else:
            self.self_model.update_from_events([event])

        if concept is not None:
            try:
                self.self_model.get_state().capability_tags.add("dummy_multimodal_alignment")
            except Exception:
                pass

    def step(self, raw_input: Optional[Any], image_path: Optional[str] = None) -> Optional[str]:
        """执行一次 Agent 的单步循环。

        流程：
            1）raw_input → perception → AgentEvent（感知事件）；
            2）事件写入 event_stream / world_model / self_model；
            3）drives.decide_intent 基于当前状态 + 最近事件给出 Intent；
            4）若 Intent 需要调用工具，则构造 ToolCall / ToolResult 事件并写入；
            5）dialogue_policy 基于 Intent + 自我描述 + 最近事件生成回复；
            6）learner.observe 观察本轮事件，为未来学习留下接口。
        """

        step_events: List[AgentEvent] = []

        # 1) 感知阶段：将外部输入转为统一事件
        if raw_input is not None:
            print(f"[感知] 收到用户输入：{raw_input}")  # noqa: T201
            perceived = self.perception.perceive(raw_input)
            perceived_events = [perceived] if isinstance(perceived, AgentEvent) else list(perceived)
            for ev in perceived_events:
                concept = self.aligner.align_event(ev) if self.aligner is not None else None
                step_events.append(ev)
                self._register_event(ev, concept)

        if image_path is not None and self.image_perception is not None:
            try:
                image_events = self.image_perception.perceive(image_path)
            except Exception as exc:  # pragma: no cover - 兼容 demo 输入错误
                print(f"[感知] 解析图片路径时出错: {exc}")  # noqa: T201
                image_events = []
            image_events_list = (
                [image_events] if isinstance(image_events, AgentEvent) else list(image_events)
            )
            for ev in image_events_list:
                concept = self.aligner.align_event(ev) if self.aligner is not None else None
                step_events.append(ev)
                self._register_event(ev, concept)

        # 2) 基于最近事件与当前模型，决策本轮意图
        recent_events = self.event_stream.to_list()
        intent = self.drive_system.decide_intent(
            self_model=self.self_model,
            world_model=self.world_model,
            recent_events=recent_events,
        )
        self.last_intent = intent

        print(f"[驱动力] 当前意图：{intent.kind}，原因：{intent.explanation}")  # noqa: T201

        # 3) 如有需要，调用工具执行具体行动
        tool_result: Optional[ToolResult] = None
        if intent.kind == "call_tool" and intent.target_tool:
            tool = self.tools.get(intent.target_tool)
            if tool is not None:
                # 构造工具调用结构
                call = ToolCall(
                    tool_name=tool.name,
                    arguments=dict(intent.extra.get("tool_args") or {}),
                    call_id=f"{self.agent_id}:tool:{intent.target_tool}:{len(recent_events)}",
                )
                call_event = AgentEvent.now(
                    event_type=EventKind.TOOL_CALL.value,
                    payload={
                        "kind": EventKind.TOOL_CALL.value,
                        "tool_name": call.tool_name,
                        "arguments": call.arguments,
                    },
                    source="agent_internal",
                )
                print(
                    f"[工具] 准备调用工具 {tool.name}，参数：{call.arguments}"  # noqa: T201
                )
                step_events.append(call_event)
                self._register_event(call_event, None)

                tool_result = tool.call(call)
                self.last_tool_result = tool_result
                result_event = AgentEvent.now(
                    event_type=EventKind.TOOL_RESULT.value,
                    payload={
                        "kind": EventKind.TOOL_RESULT.value,
                        "tool_name": tool.name,
                        "success": tool_result.success,
                    },
                    source="tool",
                )
                print(f"[工具] 工具 {tool.name} 返回：{tool_result.output}")  # noqa: T201
                step_events.append(result_event)
                self._register_event(result_event, None)
            else:
                print(  # noqa: T201
                    f"[工具] 未找到名为 {intent.target_tool} 的工具，将退回为普通回复。"
                )

        # 4) 学习阶段：观察本轮产生的事件
        if step_events:
            self.learner.observe(step_events)
            # 当前 SimpleLearner 不会主动修改模型，仅作为扩展接口存在
            self.learner.update_models(
                world_model=self.world_model,
                self_model=self.self_model,
                drive_system=self.drive_system,
                tools=self.tools,
            )

        # 5) 由对话策略根据意图生成最终回复
        reply = self.dialogue_policy.generate_reply(
            intent=intent,
            self_model=self.self_model,
            recent_events=self.event_stream.to_list(),
        )

        if reply:
            print(f"[对话] 生成回复：{reply}")  # noqa: T201
        else:
            print("[对话] 本轮选择保持沉默。")  # noqa: T201

        return reply
