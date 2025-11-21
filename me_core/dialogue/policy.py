from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from me_core.types import AgentEvent

from ..drives.base import Intent
from ..self_model.base import BaseSelfModel

if TYPE_CHECKING:
    from me_core.learning import SimpleLearner
    from me_core.world_model import SimpleWorldModel


class BaseDialoguePolicy(ABC):
    """对话策略基类。

    职责：
        - 将当前意图 + 自我描述 + 最近事件，转换为自然语言回复；
        - 不直接决定“是否开口”，该决策由 drives 层的 Intent 表达。
    """

    @abstractmethod
    def generate_reply(
        self,
        events: List[AgentEvent],
        intent: Intent,
        world: "SimpleWorldModel",
        self_model: BaseSelfModel,
        learner: "SimpleLearner",
    ) -> str | None:
        """根据意图生成一条中文回复。

        若返回 None 或空字符串，则视为本轮选择保持沉默。
        """


class RuleBasedDialoguePolicy(BaseDialoguePolicy):
    """规则驱动的对话策略实现。

    当前策略非常朴素，但完整地体现了：
        - 「我想」：为什么产生这个意图（来自 Intent.explanation）；
        - 「我要」：接下来打算以什么方式回应；
        - 「我做」：结合自我模型给出的实际回复内容。
    """

    def generate_reply(
        self,
        events: List[AgentEvent],
        intent: Intent,
        world: "SimpleWorldModel",
        self_model: BaseSelfModel,
        learner: "SimpleLearner",
    ) -> str | None:
        # stay_silent 默认不回复
        if intent.kind == "stay_silent":
            return None

        # 从最近事件中抽取最新文本
        last_user_text = ""
        for e in reversed(events):
            payload = e.payload or {}
            raw = payload.get("raw")
            if isinstance(raw, dict):
                text = raw.get("text")
                if isinstance(text, str) and text.strip():
                    last_user_text = text.strip()
                    break

        self_desc = self_model.describe_self(world_model=world)

        if intent.kind == "reflect_self":
            return self_model.describe_self(world_model=world)

        if intent.kind == "inspect_world":
            recent = world.recent_events(5)
            concept_lines: List[str] = []
            top_concepts = world.top_concepts(3)
            if top_concepts:
                concept_lines.append(
                    "概念热点：" + "；".join(f"{c.name}({s.count})" for c, s in top_concepts)
                )
            events_line = "最近事件：" + "；".join(
                f"{t.step}:{t.event.event_type}" for t in recent
            ) if recent else "最近事件：暂无"
            return f"我想回顾一下外部信息。{events_line}。" + (" ".join(concept_lines) if concept_lines else "")

        if intent.kind == "curiosity":
            target = intent.extra.get("concept_name") if intent.extra else None
            preferred = intent.preferred_modality or "更多信息"
            return f"我对「{target or '这个概念'}」很好奇，想获取{preferred}，以后有机会再进一步理解。"

        parts: list[str] = []

        if last_user_text:
            parts.append(f"你刚才说：{last_user_text}")

        if intent.explanation:
            parts.append(f"【我想】{intent.explanation}")
        else:
            parts.append("【我想】结合世界与自我状态做出回应。")

        if intent.kind == "call_tool" and intent.target_tool:
            tool_stats = learner.tool_stats.get(intent.target_tool)
            hint = ""
            if tool_stats and tool_stats.call_count:
                rate = tool_stats.success_count / tool_stats.call_count if tool_stats.call_count else 0.0
                hint = f"（历史成功率约 {rate:.0%}）"
            parts.append(f"【我要】尝试调用工具「{intent.target_tool}」{hint}，再把结果告诉你。")
        else:
            parts.append("【我要】直接回答你的问题。")

        parts.append(f"【我做】{self_desc}")
        return " ".join(parts)
