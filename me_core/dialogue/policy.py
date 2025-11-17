from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from me_core.types import AgentEvent

from ..drives.base import Intent
from ..self_model.base import BaseSelfModel


class BaseDialoguePolicy(ABC):
    """对话策略基类。

    职责：
        - 将当前意图 + 自我描述 + 最近事件，转换为自然语言回复；
        - 不直接决定“是否开口”，该决策由 drives 层的 Intent 表达。
    """

    @abstractmethod
    def generate_reply(
        self,
        intent: Intent,
        self_model: BaseSelfModel,
        recent_events: List[AgentEvent],
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
        intent: Intent,
        self_model: BaseSelfModel,
        recent_events: List[AgentEvent],
    ) -> str | None:
        # Idle / reflect 意图默认不回复，除非 extra 中显式要求
        if intent.kind in {"idle", "reflect"} and not intent.extra.get(
            "force_reply"
        ):
            return None

        self_desc = self_model.describe()

        # 从最近事件中抽取一条最新的用户文本，便于“复述用户说了什么”
        last_user_text = ""
        for e in reversed(recent_events):
            payload = e.payload or {}
            raw = payload.get("raw")
            if isinstance(raw, dict):
                text = raw.get("text")
                if isinstance(text, str) and text.strip():
                    last_user_text = text.strip()
                    break

        parts: list[str] = []

        if last_user_text:
            parts.append(f"你刚才说：{last_user_text}")

        if intent.explanation:
            parts.append(f"【我想】{intent.explanation}")
        else:
            parts.append("【我想】理解你的输入，并结合自己的状态做出回应。")

        if intent.kind == "call_tool" and intent.target_tool:
            parts.append(
                f"【我要】尝试调用工具「{intent.target_tool}」，帮助我更好地回答或行动。"
            )
        elif intent.kind == "reflect":
            parts.append("【我要】先进行一点自我反思，再看接下来该怎么做。")
        else:
            parts.append("【我要】直接根据当前的理解来回复你。")

        parts.append(f"【我做】{self_desc}")

        return " ".join(parts)

