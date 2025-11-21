from __future__ import annotations

from typing import Any, List, Protocol

from me_core.types import AgentEvent


class BasePerception(Protocol):
    """感知接口协议。

    约定：
        - 输入原始信号（文本/路径/结构化数据）；
        - 输出一组 `AgentEvent`，每个代表一个感知到的片段或样本；
        - 不负责深度语义，只做基础拆分与封装。
    """

    def perceive(self, raw_input: Any, **kwargs) -> List[AgentEvent]:
        """将原始输入转换为一组 AgentEvent。"""
        ...

