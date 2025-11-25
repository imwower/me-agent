from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Any, Optional

from me_core.types import AgentEvent

from ..drives.base import Intent
from ..self_model.base import BaseSelfModel

if TYPE_CHECKING:
    from me_core.learning import SimpleLearner
    from me_core.world_model import SimpleWorldModel

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        policy_config: Any = None,
        agent_config: Any | None = None,
        dialogue_llm: Any | None = None,
    ) -> None:
        self.policy_config = policy_config
        self.agent_config = agent_config
        self.dialogue_llm = dialogue_llm
        self.use_llm_dialogue = bool(getattr(agent_config, "use_llm_dialogue", False))
        self.max_llm_reply_length = 400

    def _build_llm_prompt(
        self,
        events: List[AgentEvent],
        intent: Intent,
        world: "SimpleWorldModel",
        self_model: BaseSelfModel,
    ) -> str:
        recent_texts: List[str] = []
        for e in reversed(events[-5:]):
            payload = e.payload or {}
            raw = payload.get("raw") if isinstance(payload, dict) else None
            if isinstance(raw, dict) and isinstance(raw.get("text"), str):
                recent_texts.append(raw["text"])
        self_desc = self_model.describe_self(world_model=world, max_concepts=3)
        world_summary = world.summarize()
        world_str = json.dumps(world_summary, ensure_ascii=False)
        if len(world_str) > 500:
            world_str = world_str[:500] + "..."
        intent_desc = intent.explanation or intent.message or intent.kind
        prompt = (
            "你是一个受限的对话助手，需要根据给定的内部状态生成中文回复。"
            "请保持简洁，并避免编造无关细节。\n"
            f"【意图】{intent.kind}: {intent_desc}\n"
            f"【自我】{self_desc}\n"
            f"【世界】{world_str}\n"
            f"【最近输入】{'; '.join(reversed(recent_texts)) if recent_texts else '无'}\n"
            "请输出一句中文回复。"
        )
        return prompt

    def _safe_reply_text(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        reply = str(text).strip()
        if not reply:
            return None
        if len(reply) > self.max_llm_reply_length:
            reply = reply[: self.max_llm_reply_length] + "..."
        return reply

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

        max_concepts = 3
        if self.policy_config and getattr(self.policy_config, "dialogue", None):
            try:
                max_concepts = int(self.policy_config.dialogue.max_recent_concepts)
            except Exception:
                pass
        self_desc = self_model.describe_self(world_model=world, max_concepts=max_concepts)
        state = self_model.get_state()
        brain_mode = getattr(state, "last_brain_mode", "unknown")
        brain_conf = float(getattr(state, "last_brain_confidence", 0.0) or 0.0)
        brain_hint = ""
        if brain_mode and brain_mode != "unknown":
            brain_hint = f"【脑态】当前内部脑模式偏向{brain_mode}，信心约 {brain_conf:.2f}。"

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

        if self.use_llm_dialogue and self.dialogue_llm:
            prompt = self._build_llm_prompt(events, intent, world, self_model)
            try:
                llm_reply = self.dialogue_llm.generate_reply(prompt, meta={"intent": intent.kind})
            except Exception as exc:  # pragma: no cover - 外部调用
                logger.warning("LLM 对话失败，回退到规则模板: %s", exc)
                llm_reply = None
            safe_reply = self._safe_reply_text(llm_reply)
            if safe_reply:
                return safe_reply

        if intent.kind == "curiosity":
            target = intent.extra.get("concept_name") if intent.extra else None
            preferred = intent.preferred_modality or "更多信息"
            return f"我对「{target or '这个概念'}」很好奇，想获取{preferred}，以后有机会再进一步理解。{brain_hint}"

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

        if brain_hint:
            parts.append(brain_hint)
        parts.append(f"【我做】{self_desc}")
        return " ".join(parts)
