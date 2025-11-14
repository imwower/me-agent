from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DrivesConfig:
    """驱动力相关的可调参数配置。

    这些参数在整个项目中用于：
        - 用户显式指令（apply_user_command）的增量步长；
        - 隐式反馈（implicit_adjust）的平滑系数与阈值。
    """

    # 显式用户指令调整步长，例如 “多陪我聊天” 每次提升多少
    user_command_step: float = 0.2

    # 隐式平滑更新系数：new = old * (1 - alpha) + target * alpha
    implicit_smooth_alpha: float = 0.1

    # 用户响应高/低的阈值
    high_response_threshold: float = 0.6
    low_response_threshold: float = 0.3

    # 探索值被视为“偏高”的阈值
    exploration_high_threshold: float = 0.6

    # 学习成功率被视为“较高”的阈值
    learning_success_high_threshold: float = 0.7

    # 在学习结果驱动下调整探索/学习强度所使用的基础步长
    learning_adjust_step: float = 0.2


DEFAULT_DRIVES_CONFIG = DrivesConfig()

