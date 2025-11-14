from __future__ import annotations

import logging
from typing import Dict

from .planner import InitiativeDecision

logger = logging.getLogger(__name__)


def generate_message(
    decision: InitiativeDecision,
    self_summary: Dict[str, str],
    context: Dict[str, object],
) -> str:
    """根据对话决策与自我总结，生成一段中文输出。

    当前为桩实现，可以在未来替换为真正的语言模型。
    """

    if not decision.should_speak or decision.intent == "silent":
        logger.info("决策为保持沉默，不生成文本。")
        return ""

    who = (self_summary.get("who_am_i") or "").strip()
    can = (self_summary.get("what_can_i_do") or "").strip()
    need = (self_summary.get("what_do_i_need") or "").strip()
    topic = decision.topic or str(context.get("topic") or "当前话题")

    logger.info(
        "生成对话文本: intent=%s, topic=%r, who=%r",
        decision.intent,
        topic,
        who,
    )

    if decision.intent == "self_introduction":
        parts = []
        if who:
            parts.append(who)
        if can:
            parts.append(f"简单来说，{can}")
        if need:
            parts.append(f"目前，我觉得{need}")
        else:
            parts.append("目前我会根据你的任务和反馈持续调整自己。")
        message = " ".join(parts)
    elif decision.intent == "ask_for_help":
        base = who or "我是一个正在学习中的智能体。"
        need_part = need or "我目前在数据和反馈上还有很多不足。"
        message = (
            f"{base} 现在，我想向你说明一下：{need_part} "
            f"如果你愿意，可以在「{topic}」或相关方向给我一些任务或反馈，"
            "帮助我更好地改进自己。"
        )
    elif decision.intent == "share_learning":
        base = who or "我是一个正在尝试自我改进的智能体。"
        can_part = can or "我正在学习如何更系统地理解自己的能力与局限。"
        message = (
            f"{base} 最近，我在围绕「{topic}」做一些学习和实验，"
            f"{can_part} 如果你有想法，也可以一起探索。"
        )
    else:
        # 未知意图时给出一个通用自述
        message = (
            who
            or "你好，我是一个正在逐步构建自我模型和内在驱动力的智能体。"
        )

    logger.info("生成的对话文本: %s", message)
    return message

