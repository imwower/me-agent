from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, Mapping

from .config import DEFAULT_DRIVES_CONFIG
from .drive_vector import DriveVector

logger = logging.getLogger(__name__)

# 为了后续调参方便，这里统一通过配置对象读取相关超参数
CONFIG = DEFAULT_DRIVES_CONFIG


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

    step = CONFIG.user_command_step

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

    # 处理用户响应比例：根据用户回复情况，逐步提升或降低聊天/社交倾向
    if isinstance(user_response_ratio_raw, (int, float)):
        user_response_ratio = float(user_response_ratio_raw)
        # 将反馈裁剪在 [0,1] 内，避免异常值
        if user_response_ratio < 0.0:
            user_response_ratio = 0.0
        elif user_response_ratio > 1.0:
            user_response_ratio = 1.0

        # 为了避免“只升不降”的单向偏移，这里根据不同区间采用不同策略：
        # - 高响应（>=0.6）：鼓励多说话，目标向 user_response_ratio 靠近；
        # - 低响应（<=0.3）：说明用户并不买账，适当降低话痨度和社交需求；
        # - 中等响应：维持在一个相对中性的水平（0.5），避免极端。
        if user_response_ratio >= CONFIG.high_response_threshold:
            target_chat = max(new_drives.chat_level, user_response_ratio)
            target_social = max(new_drives.social_need, user_response_ratio)
            logger.info(
                "用户响应较高，适度提升聊天/社交倾向: ratio=%.3f, target_chat=%.3f, target_social=%.3f",
                user_response_ratio,
                target_chat,
                target_social,
            )
        elif user_response_ratio <= CONFIG.low_response_threshold:
            target_chat = min(new_drives.chat_level, user_response_ratio)
            target_social = min(new_drives.social_need, user_response_ratio)
            logger.info(
                "用户响应较低，适度降低聊天/社交倾向: ratio=%.3f, target_chat=%.3f, target_social=%.3f",
                user_response_ratio,
                target_chat,
                target_social,
            )
        else:
            target_chat = 0.5
            target_social = 0.5
            logger.info(
                "用户响应中等，将聊天/社交倾向缓慢拉回中性水平: ratio=%.3f",
                user_response_ratio,
            )

        # 使用平滑更新公式：new = old * (1 - alpha) + target * alpha
        alpha = CONFIG.implicit_smooth_alpha
        new_drives.chat_level = (
            new_drives.chat_level * (1.0 - alpha) + target_chat * alpha
        )
        new_drives.social_need = (
            new_drives.social_need * (1.0 - alpha) + target_social * alpha
        )

    # 处理学习成功率：根据近期“试验”的成败，略微调节探索/学习倾向
    if isinstance(learning_success_raw, (int, float)):
        learning_success = float(learning_success_raw)
        if learning_success < 0.0:
            learning_success = 0.0
        elif learning_success > 1.0:
            learning_success = 1.0

        if learning_success == 0.0 and new_drives.exploration_level > CONFIG.exploration_high_threshold:
            # 近期尝试全部失败且探索值偏高：略微降低探索欲与学习强度
            target_exploration = max(
                0.0, new_drives.exploration_level - CONFIG.learning_adjust_step
            )
            alpha = CONFIG.implicit_smooth_alpha
            new_drives.exploration_level = (
                new_drives.exploration_level * (1.0 - alpha)
                + target_exploration * alpha
            )

            target_intensity = max(
                0.0, new_drives.learning_intensity - CONFIG.learning_adjust_step
            )
            alpha = CONFIG.implicit_smooth_alpha
            new_drives.learning_intensity = (
                new_drives.learning_intensity * (1.0 - alpha)
                + target_intensity * alpha
            )
            logger.info(
                "近期学习完全失败，轻微压低探索欲和学习强度: exploration=%.3f, learning_intensity=%.3f",
                new_drives.exploration_level,
                new_drives.learning_intensity,
            )
        elif learning_success >= CONFIG.learning_success_high_threshold:
            # 若近期学习成功率较高，则在不过度冒进的前提下，小幅提升学习强度
            target_intensity = min(
                1.0, new_drives.learning_intensity + CONFIG.learning_adjust_step
            )
            alpha = CONFIG.implicit_smooth_alpha
            new_drives.learning_intensity = (
                new_drives.learning_intensity * (1.0 - alpha)
                + target_intensity * alpha
            )
            logger.info(
                "近期学习较成功，适度提升学习强度: learning_success=%.3f, learning_intensity=%.3f",
                learning_success,
                new_drives.learning_intensity,
            )

    new_drives.clamp()

    logger.info("隐式调整后的驱动力: %s", new_drives)
    return new_drives


def apply_baseline_homeostasis(
    drives: DriveVector,
    baseline: Mapping[str, float],
    alpha: float = 0.1,
) -> DriveVector:
    """根据给定的基线驱动力，让当前驱动缓慢回归“默认水平”。

    设计意图：
        - 驱动力在长期运行中会受到各种显式/隐式反馈的影响；
        - 为避免参数漂移到极端值，这里提供一个“向基线缓慢回归”的机制；
        - 使用简单的一阶平滑公式：
              new = old * (1 - alpha) + baseline * alpha
          其中 alpha ∈ (0,1)，越大表示回归越快。

    参数：
        drives: 当前驱动力向量；
        baseline: 各字段的基线值字典，例如 {"chat_level": 0.5, ...}；
        alpha: 回归速率系数，默认 0.1。

    返回：
        一个新的 DriveVector 实例，表示回归后的驱动力。
    """

    if alpha <= 0.0:
        # alpha 为 0 时不进行任何回归，直接返回原驱动力的拷贝
        logger.info("alpha <= 0，跳过基线回归，直接返回原驱动力副本。")
        return replace(drives)

    new_drives = replace(drives)

    logger.info(
        "应用基线稳态前的驱动力: %s, baseline=%s, alpha=%.3f",
        drives,
        baseline,
        alpha,
    )

    for field_name in (
        "chat_level",
        "curiosity_level",
        "exploration_level",
        "learning_intensity",
        "social_need",
        "data_need",
    ):
        old_value = getattr(new_drives, field_name)
        base_value_raw = baseline.get(field_name, old_value)
        try:
            base_value = float(base_value_raw)
        except (TypeError, ValueError):
            base_value = old_value

        new_value = old_value * (1.0 - alpha) + base_value * alpha
        setattr(new_drives, field_name, new_value)

    new_drives.clamp()

    logger.info("应用基线稳态后的驱动力: %s", new_drives)
    return new_drives
