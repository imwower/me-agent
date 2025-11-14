from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict

from .drive_vector import DriveVector

logger = logging.getLogger(__name__)


def _increment(value: float, delta: float) -> float:
    """简单的加法更新工具函数，便于后续调整策略。

    该函数本身不做裁剪，裁剪逻辑由 DriveVector.clamp 统一处理。
    """

    return value + delta


def apply_user_command(drives: DriveVector, command: str) -> DriveVector:
    """根据用户的高层指令，显式地调整驱动力。

    这里采用非常直观的规则映射：
    - “陪我聊会”/“多陪我聊天”：提高 chat_level, social_need
    - “今天先安静点”：降低 chat_level, social_need
    - “今天多探索点新东西”：提高 curiosity_level, exploration_level
    - “先稳一点别乱折腾”：降低 exploration_level, learning_intensity

    每次调整采用固定增量（例如 ±0.2），之后调用 clamp 限制在 [0,1]。
    为了避免副作用，该函数会在原对象基础上复制出一个新的 DriveVector 返回。
    """

    normalized = command.strip()
    new_drives = replace(drives)  # 创建一个浅拷贝，避免直接修改原对象

    step = 0.2

    logger.info("应用用户指令前的驱动力: %s, 指令: %s", drives, normalized)

    # 根据用户希望“多聊天”的指令，增强聊天与社交相关驱动
    if "陪我聊" in normalized or "聊天" in normalized:
        new_drives.chat_level = _increment(new_drives.chat_level, step)
        new_drives.social_need = _increment(new_drives.social_need, step)

    # 用户希望“安静一点”，减少聊天欲望与社交需求
    if "安静" in normalized:
        new_drives.chat_level = _increment(new_drives.chat_level, -step)
        new_drives.social_need = _increment(new_drives.social_need, -step)

    # 用户希望“多探索新东西”，增加好奇心与探索欲
    if "探索" in normalized or "新东西" in normalized:
        new_drives.curiosity_level = _increment(new_drives.curiosity_level, step)
        new_drives.exploration_level = _increment(
            new_drives.exploration_level, step
        )

    # 用户希望“稳一点别乱折腾”，降低探索欲与学习强度
    if "稳" in normalized or "折腾" in normalized:
        new_drives.exploration_level = _increment(
            new_drives.exploration_level, -step
        )
        new_drives.learning_intensity = _increment(
            new_drives.learning_intensity, -step
        )

    # 统一裁剪到合法范围
    new_drives.clamp()

    logger.info("应用用户指令后的驱动力: %s", new_drives)
    return new_drives


def implicit_adjust(drives: DriveVector, feedback: Dict[str, Any]) -> DriveVector:
    """根据隐式反馈统计，平滑调整驱动力。

    反馈示例：
    - user_response_ratio: 0~1，表示用户对当前对话/行为的响应程度
    - learning_success: 0~1，表示近期学习/探索行为的成功率

    更新策略示意：
    - 如果 user_response_ratio 较高，则希望逐步提升 chat_level / social_need；
      采用“旧值 * 0.9 + 目标值 * 0.1”的方式，保证变化是缓慢的。
    - 如果 learning_success 为 0 且 exploration_level 较高，则略微降低探索欲，
      避免在完全没有收益的情况下持续“乱折腾”。
    """

    new_drives = replace(drives)

    user_response_ratio_raw = feedback.get("user_response_ratio")
    learning_success_raw = feedback.get("learning_success")

    logger.info("隐式调整前的驱动力: %s, 反馈: %s", drives, feedback)

    # 处理用户响应比例：用户回复越积极，越鼓励聊天与社交
    if isinstance(user_response_ratio_raw, (int, float)):
        user_response_ratio = float(user_response_ratio_raw)
        # 将反馈裁剪在 [0,1] 内，避免异常值
        if user_response_ratio < 0.0:
            user_response_ratio = 0.0
        elif user_response_ratio > 1.0:
            user_response_ratio = 1.0

        # 目标值倾向于不低于当前值：高响应会把目标往上拉
        target_chat = max(new_drives.chat_level, user_response_ratio)
        target_social = max(new_drives.social_need, user_response_ratio)

        # 使用平滑更新公式：new = old * 0.9 + target * 0.1
        new_drives.chat_level = (
            new_drives.chat_level * 0.9 + target_chat * 0.1
        )
        new_drives.social_need = (
            new_drives.social_need * 0.9 + target_social * 0.1
        )

    # 处理学习成功率：若探索欲很高但完全没有成功，则轻微压低探索欲
    if isinstance(learning_success_raw, (int, float)):
        learning_success = float(learning_success_raw)
        if learning_success < 0.0:
            learning_success = 0.0
        elif learning_success > 1.0:
            learning_success = 1.0

        if learning_success == 0.0 and new_drives.exploration_level > 0.6:
            # 目标探索值略低于当前值，避免一次性大幅削弱
            target_exploration = max(
                0.0, new_drives.exploration_level - 0.2
            )
            new_drives.exploration_level = (
                new_drives.exploration_level * 0.9
                + target_exploration * 0.1
            )

    new_drives.clamp()

    logger.info("隐式调整后的驱动力: %s", new_drives)
    return new_drives

