from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from me_core.event_stream import EventStream
from me_core.types import AgentEvent, EventKind, ToolCall, ToolResult

from ..dialogue import BaseDialoguePolicy
from ..drives import BaseDriveSystem, Intent
from ..learning import BaseLearner
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

    def step(self, raw_input: Optional[Any]) -> Optional[str]:  # pragma: no cover - 接口定义
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
    # 为后续“种群进化 / 多教师模式”等扩展预留标识字段
    agent_id: str = "default"
    population_id: Optional[str] = None

    # 便于调试与测试的最后一轮内部状态记录
    last_intent: Optional[Intent] = field(default=None, init=False)
    last_tool_result: Optional[ToolResult] = field(default=None, init=False)

    def _append_events(self, *events: AgentEvent) -> None:
        """将事件写入事件流并更新 world_model / self_model。"""

        if not events:
            return
        for e in events:
            # 补充 agent_id / population_id 等元信息，方便后续分析
            e.meta.setdefault("agent_id", self.agent_id)
            if self.population_id is not None:
                e.meta.setdefault("population_id", self.population_id)
            self.event_stream.append_event(e)
            self.event_stream.log_event(e)
        self.world_model.update(list(events))
        self.self_model.update_from_events(list(events))

    def step(self, raw_input: Optional[Any]) -> Optional[str]:
        """执行一次 Agent 的单步循环。

        流程：
            1）raw_input → perception → AgentEvent（感知事件）；
            2）事件写入 event_stream / world_model / self_model；
            3）drives.decide_intent 基于当前状态 + 最近事件给出 Intent；
            4）若 Intent 需要调用工具，则构造 ToolCall / ToolResult 事件并写入；
            5）dialogue_policy 基于 Intent + 自我描述 + 最近事件生成回复；
            6）learner.observe 观察本轮事件，为未来学习留下接口。
        """

        step_events: list[AgentEvent] = []

        # 1) 感知阶段：将外部输入转为统一事件
        if raw_input is not None:
            print(f"[感知] 收到用户输入：{raw_input}")  # noqa: T201
            perception_event = self.perception.perceive(raw_input)
            step_events.append(perception_event)
            self._append_events(perception_event)

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
                self._append_events(call_event)

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
                print(
                    f"[工具] 工具 {tool.name} 返回：{tool_result.output}"  # noqa: T201
                )
                step_events.append(result_event)
                self._append_events(result_event)
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

