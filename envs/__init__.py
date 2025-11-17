"""封闭环境（envs）相关模块。

当前阶段提供：
- BaseEnv: 环境接口定义（reset/step/get_primitives）；
- GridWorldEnv: 简化的二维网格环境实现，用于早期自监督与探索实验。
"""

from .core_env import BaseEnv  # noqa: F401
from .gridworld import GridWorldEnv  # noqa: F401

__all__ = [
    "BaseEnv",
    "GridWorldEnv",
]

