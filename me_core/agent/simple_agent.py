from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List

from me_core.event_stream import EventStream
from me_core.types import AgentEvent, EventKind, ToolCall, ToolResult

from me_core.alignment.concepts import ConceptSpace
from me_core.alignment.embeddings import DummyEmbeddingBackend, create_embedding_backend
from me_core.alignment.aligner import MultimodalAligner

from me_core.config import AgentConfig
from ..dialogue import BaseDialoguePolicy
from ..drives import BaseDriveSystem, Intent
from ..learning import BaseLearner
from ..memory import EpisodicMemory, SemanticMemory, JsonlMemoryStorage
from ..introspection import IntrospectionGenerator, IntrospectionLog
from ..perception import ImagePerception
from ..perception.base import BasePerception
from ..self_model import BaseSelfModel
from ..tools import BaseTool
from ..world_model import BaseWorldModel
import me_core.logging as agent_logging


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
    config: Optional[AgentConfig] = field(default=None, repr=False)
    episodic_memory: Optional[EpisodicMemory] = field(default=None, repr=False)
    semantic_memory: Optional[SemanticMemory] = field(default=None, repr=False)
    introspection_generator: Optional[IntrospectionGenerator] = field(default=None, repr=False)
    logger: Optional[logging.Logger] = field(default=None, repr=False)
    timeline_path: Optional[Path] = field(default=None, repr=False)
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
    _local_step: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config is not None:
            if self.timeline_path is None and self.config.timeline_path:
                self.timeline_path = Path(self.config.timeline_path)
            # 驱动力开关
            if hasattr(self.drive_system, "enable_curiosity"):
                try:
                    self.drive_system.enable_curiosity = bool(self.config.enable_curiosity)
                except Exception:
                    pass
            if hasattr(self.drive_system, "enable_reflection"):
                try:
                    self.drive_system.enable_reflection = bool(self.config.enable_introspection)
                except Exception:
                    pass

        # 根据配置选择 embedding backend
        try:
            self.embedding_backend = create_embedding_backend(self.config)
        except Exception:
            self.embedding_backend = DummyEmbeddingBackend(dim=64)

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
            if hasattr(self.self_model, "register_capability_tag"):
                self.self_model.register_capability_tag("dummy_multimodal_alignment")  # type: ignore[attr-defined]
            else:
                self.self_model.get_state().capability_tags.add("dummy_multimodal_alignment")
        except Exception:
            pass

        if self.logger is None:
            try:
                self.logger = agent_logging.setup_logger()
            except Exception:
                self.logger = None

        if self.timeline_path is not None and not isinstance(self.timeline_path, Path):
            self.timeline_path = Path(self.timeline_path)  # type: ignore[assignment]

        # 初始化持久化记忆
        if self.episodic_memory is None or self.semantic_memory is None:
            episodes_path = getattr(self.config, "episodes_path", None) if self.config else None
            concepts_path = getattr(self.config, "concepts_path", None) if self.config else None
            if episodes_path:
                storage = JsonlMemoryStorage(Path(episodes_path), Path(concepts_path) if concepts_path else None)
                if self.episodic_memory is None:
                    self.episodic_memory = EpisodicMemory(storage)
                if self.semantic_memory is None:
                    self.semantic_memory = SemanticMemory(storage)

        if self.introspection_generator is None:
            try:
                self.introspection_generator = IntrospectionGenerator(
                    world=self.world_model, self_model=self.self_model, learner=self.learner  # type: ignore[arg-type]
                )
            except Exception:
                self.introspection_generator = None

    def _next_step(self) -> int:
        """获取下一个 step 编号。"""

        advancer = getattr(self.world_model, "advance_step", None)
        if callable(advancer):
            try:
                return int(advancer())
            except Exception:
                pass
        self._local_step += 1
        return self._local_step

    def _register_event(self, event: AgentEvent, step: int, concept: Optional[Any] = None) -> None:
        """集中处理事件写入、世界/自我模型更新与日志。"""

        event.meta.setdefault("agent_id", self.agent_id)
        if self.population_id is not None:
            event.meta.setdefault("population_id", self.population_id)
        event.meta.setdefault("step", step)

        if hasattr(self.world_model, "append_event"):
            self.world_model.append_event(event)  # type: ignore[attr-defined]
        elif hasattr(self.world_model, "observe_event"):
            self.world_model.observe_event(event, concept)  # type: ignore[attr-defined]

        if hasattr(self.self_model, "observe_event"):
            self.self_model.observe_event(event, step=step)  # type: ignore[arg-type]
        else:
            try:
                self.self_model.update_from_events([event], step=step)  # type: ignore[arg-type]
            except TypeError:
                self.self_model.update_from_events([event])  # type: ignore[arg-type]

        self.event_stream.append_event(event)
        if self.timeline_path is not None:
            try:
                with Path(self.timeline_path).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            except Exception:
                pass
        if self.logger:
            try:
                self.logger.debug(event.pretty())
            except Exception:
                pass
        else:
            self.event_stream.log_event(event)

        if concept is not None:
            try:
                if hasattr(self.self_model, "register_capability_tag"):
                    self.self_model.register_capability_tag("dummy_multimodal_alignment")  # type: ignore[attr-defined]
                else:
                    self.self_model.get_state().capability_tags.add("dummy_multimodal_alignment")
            except Exception:
                pass

    def step(
        self,
        raw_input: Optional[Any],
        image_path: Optional[str] = None,
        debug: bool = False,
    ) -> Optional[str]:
        """执行一次 Agent 的单步循环。"""

        step_id = self._next_step()
        step_events: List[AgentEvent] = []

        # 1) world_model 已前进一步，感知阶段
        if raw_input is not None:
            if self.logger:
                self.logger.info("[感知] 收到输入: %s", raw_input)
            perceived = self.perception.perceive(raw_input)
            perceived_events = [perceived] if isinstance(perceived, AgentEvent) else list(perceived)
            for ev in perceived_events:
                concept = self.aligner.align_event(ev) if self.aligner is not None else None
                step_events.append(ev)
                self._register_event(ev, step_id, concept)

        if image_path is not None and self.image_perception is not None:
            try:
                image_events = self.image_perception.perceive(image_path)
            except Exception as exc:  # pragma: no cover - demo 兼容路径错误
                if self.logger:
                    self.logger.warning("解析图片路径时出错: %s", exc)
                image_events = []
            image_events_list = (
                [image_events] if isinstance(image_events, AgentEvent) else list(image_events)
            )
            for ev in image_events_list:
                concept = self.aligner.align_event(ev) if self.aligner is not None else None
                step_events.append(ev)
                self._register_event(ev, step_id, concept)

        # 2) 决策本轮意图
        recent_events = self.event_stream.to_list()
        intent = self.drive_system.decide_intent(
            self_model=self.self_model,
            world_model=self.world_model,
            recent_events=recent_events,
        )
        self.last_intent = intent
        if debug and self.logger:
            self.logger.info("[驱动力] intent=%s priority=%s msg=%s", intent.kind, intent.priority, intent.message or intent.explanation)

        # 3) 执行动作（工具调用等）
        tool_result: Optional[ToolResult] = None
        tool_success: Optional[bool] = None
        if intent.kind == "call_tool" and intent.target_tool:
            tool = self.tools.get(intent.target_tool)
            if tool is not None:
                call = ToolCall(
                    tool_name=getattr(tool, "name", intent.target_tool),
                    arguments=dict(intent.extra.get("tool_args") or {}),
                    call_id=f"{self.agent_id}:tool:{intent.target_tool}:{step_id}",
                )
                call_event = AgentEvent.now(
                    event_type=EventKind.TOOL_CALL.value,
                    payload={
                        "kind": EventKind.TOOL_CALL.value,
                        "tool_name": call.tool_name,
                        "arguments": call.arguments,
                    },
                    source="agent_internal",
                    kind=EventKind.TOOL_CALL,
                )
                step_events.append(call_event)
                self._register_event(call_event, step_id, None)

                try:
                    output = tool.run(call.arguments)  # type: ignore[attr-defined]
                    tool_success = True
                    error_msg = None
                except Exception as exc:  # pragma: no cover - 依赖外部环境的调用异常
                    output = {"tool_name": getattr(tool, "name", intent.target_tool), "error": str(exc)}
                    tool_success = False
                    error_msg = str(exc)

                tool_result = ToolResult(
                    call_id=call.call_id,
                    success=bool(tool_success),
                    output=output,
                    error=error_msg,
                )
                self.last_tool_result = tool_result
                if hasattr(self.self_model, "register_capability_tag"):
                    self.self_model.register_capability_tag(f"{tool.name}_tool")  # type: ignore[attr-defined]

                result_event = AgentEvent.now(
                    event_type=EventKind.TOOL_RESULT.value,
                    payload={
                        "kind": EventKind.TOOL_RESULT.value,
                        "tool_name": getattr(tool, "name", intent.target_tool),
                        "success": tool_success,
                        "output": output,
                        "error": error_msg,
                    },
                    source="tool",
                    kind=EventKind.TOOL_RESULT,
                )
                step_events.append(result_event)
                self._register_event(result_event, step_id, None)
                self.learner.observe_tool_result(getattr(tool, "name", intent.target_tool), bool(tool_success))
            else:
                tool_success = False
                if self.logger:
                    self.logger.warning("未找到名为 %s 的工具，将退回为普通回复。", intent.target_tool)

        # 4) 由对话策略根据意图生成最终回复
        reply = self.dialogue_policy.generate_reply(
            events=self.event_stream.to_list(),
            intent=intent,
            world=self.world_model,
            self_model=self.self_model,
            learner=self.learner,
        )

        if reply:
            reply_event = AgentEvent.now(
                event_type=EventKind.DIALOGUE.value,
                payload={"raw": {"text": reply}, "direction": "outgoing"},
                source="agent_internal",
                kind=EventKind.DIALOGUE,
            )
            step_events.append(reply_event)
            self._register_event(reply_event, step_id, None)

        # 5) 学习：观察本轮产生的事件与意图结果
        if step_events:
            self.learner.observe(step_events)
            self.learner.update_models(
                world_model=self.world_model,
                self_model=self.self_model,
                drive_system=self.drive_system,
                tools=self.tools,
            )

        intent_success = tool_success if intent.kind == "call_tool" else True
        self.learner.observe_intent_outcome(intent, bool(intent_success))

        # 6) 写入长期记忆
        if self.episodic_memory is not None:
            episode = self.episodic_memory.begin_episode(step_id, tags=None)
            summary = self._summarize_step(step_events, intent, reply)
            self.episodic_memory.end_episode(episode, step_id, step_events, summary)
        if self.semantic_memory is not None:
            self._sync_semantic_memory()

        if debug and self.logger:
            top_concepts = getattr(self.world_model, "top_concepts", lambda top_k=3: [])(top_k=3)
            top_names = ", ".join(c.name for c, _ in top_concepts) if top_concepts else "-"
            self.logger.info(
                "[调试] step=%s intent=%s reply=%s top_concepts=%s self=%s",
                step_id,
                intent.kind,
                (reply or "无"),
                top_names,
                self.self_model.describe_self(world_model=self.world_model),
            )
            agent_logging.log_step(
                self.logger,
                step=step_id,
                intent_kind=intent.kind,
                reply=reply,
                tool_name=intent.target_tool if intent.kind == "call_tool" else None,
                tool_success=tool_success,
                events=step_events,
            )

        if reply and not self.logger:
            print(f"[对话] 生成回复：{reply}")  # noqa: T201

        return reply

    def _summarize_step(self, events: List[AgentEvent], intent: Intent, reply: Optional[str]) -> str:
        texts: List[str] = []
        for ev in events:
            payload = ev.payload or {}
            if isinstance(payload, dict):
                raw = payload.get("raw")
                if isinstance(raw, dict) and isinstance(raw.get("text"), str):
                    texts.append(raw["text"])
        intent_part = f"意图={intent.kind}"
        reply_part = f"回复={reply}" if reply else ""
        return "; ".join([intent_part] + texts + ([reply_part] if reply_part else []))

    def _sync_semantic_memory(self) -> None:
        if self.semantic_memory is None:
            return
        top_concepts = getattr(self.world_model, "top_concepts", lambda top_k=3: [])(top_k=3)
        for concept, stats in top_concepts:
            modalities = []
            if hasattr(stats, "modalities") and isinstance(stats.modalities, dict):
                modalities = [f"{k}:{v}" for k, v in stats.modalities.items()]
            desc = f"概念「{concept.name}」最近出现 {stats.count} 次，模态分布：{', '.join(modalities) or '未知'}。"
            self.semantic_memory.upsert_concept_memory(
                concept_id=concept.id,
                name=concept.name,
                description=desc,
                tags=set(modalities),
            )

    def introspect(
        self,
        scenario_id: Optional[str],
        start_step: int,
        end_step: Optional[int] = None,
        test_failures: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> Optional[IntrospectionLog]:
        if self.introspection_generator is None:
            return None
        final_step = end_step if end_step is not None else getattr(self.world_model, "current_step", start_step)
        return self.introspection_generator.generate(
            scenario_id=scenario_id, start_step=start_step, end_step=final_step, test_failures=test_failures, notes=notes
        )
