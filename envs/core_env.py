from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple


EnvObs = Any
EnvAction = Any
EnvStepReturn = Tuple[EnvObs, float, bool, Dict[str, Any]]
PrimitiveFn = Callable[..., EnvStepReturn]


class BaseEnv(ABC):
    """环境接口基类。

    设计意图：
        - 为 AgentCore 提供统一的交互接口；
        - 所有具体环境（如 GridWorldEnv）都应实现 reset / step / get_primitives；
        - 不对观测/动作的具体结构做强约束，只要求兼容序列化与日志记录。
    """

    @abstractmethod
    def reset(self) -> EnvObs:
        """重置环境到初始状态，并返回初始观测。"""

    @abstractmethod
    def step(self, action: EnvAction) -> EnvStepReturn:
        """在环境中执行一个动作，返回 (obs, reward, done, info)。"""

    @abstractmethod
    def get_primitives(self) -> Dict[str, PrimitiveFn]:
        """返回一组可供工具系统使用的“原语”动作。

        返回值为:
            名称 -> 函数
        约定：
            - 函数调用签名应与 step 基本兼容（可接受少量 context 参数）；
            - 目前阶段可以简单将原语包装为对 step 的不同动作常量调用。
        """

