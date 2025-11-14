from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from me_core.drives.drive_vector import DriveVector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InitiativeDecision:
    """描述一次对话上的“是否开口”决策。

    字段：
        should_speak: 是否主动说话
        intent: 高层意图标签，例如：
            - "self_introduction"：自我介绍 / 自我陈述
            - "ask_for_help"：向用户请求帮助或反馈
            - "share_learning"：分享最近的学习或观察
            - "silent"：保持安静，不主动输出
        topic: 该轮对话的主要主题（可选）
    """

    should_speak: bool
    intent: str
    topic: Optional[str] = None


class DialoguePlanner:
    """简单的对话规划器。

    当前实现使用启发式规则：
        - 根据 chat_level 与 social_need 估计“说话倾向”；
        - 若倾向较弱，则保持沉默；
        - 若倾向较强且有明显需要，则优先选择 ask_for_help；
        - 否则默认 self_introduction。
    """

    def decide_initiative(
        self,
        drives: DriveVector,
        self_summary: Dict[str, str],
        context: Dict[str, object],
    ) -> InitiativeDecision:
        """基于驱动力与自我总结，决定是否主动发起对话。"""

        speak_score = 0.5 * drives.chat_level + 0.5 * drives.social_need
        need_text = (self_summary.get("what_do_i_need") or "").strip()
        topic = context.get("topic")
        topic_str = str(topic) if topic is not None else None

        logger.info(
            "对话决策输入: speak_score=%.3f, data_need=%.3f, topic=%r, needs=%r",
            speak_score,
            drives.data_need,
            topic_str,
            need_text,
        )

        # 说话倾向过低时，保持沉默
        if speak_score < 0.3:
            decision = InitiativeDecision(
                should_speak=False,
                intent="silent",
                topic=topic_str,
            )
            logger.info("对话决策结果: %s", decision)
            return decision

        # 有较强的数据需求且明确表达了“需要”，则倾向于请求帮助
        if need_text and drives.data_need >= 0.6:
            decision = InitiativeDecision(
                should_speak=True,
                intent="ask_for_help",
                topic=topic_str,
            )
            logger.info("对话决策结果: %s", decision)
            return decision

        # 默认进行一次简短的自我介绍/自述
        decision = InitiativeDecision(
            should_speak=True,
            intent="self_introduction",
            topic=topic_str,
        )
        logger.info("对话决策结果: %s", decision)
        return decision

