from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolInfo:
    """工具的元信息描述。

    字段：
        name: 工具名称（唯一标识）
        type: 工具类型，例如 "knowledge" / "simulation" / "logs" 等
        cost: 抽象成本 [0,1]，综合时间、资源等（越高越“贵”）
        description: 工具用途的简短说明
        good_for: 适合处理的主题或问题类型的关键词列表
    """

    name: str
    type: str
    cost: float
    description: str
    good_for: List[str] = field(default_factory=list)


class ToolRegistry:
    """简单的工具注册表，用于按主题检索可用工具。"""

    def __init__(self) -> None:
        # 使用名称作为键，方便覆盖更新
        self._tools: Dict[str, ToolInfo] = {}

    def register_tool(self, info: ToolInfo) -> None:
        """注册一个工具，若名称重复则覆盖旧记录。"""

        logger.info("注册工具: %s", info)
        self._tools[info.name] = info

    def list_tools(self) -> List[ToolInfo]:
        """返回当前已注册的全部工具列表。"""

        return list(self._tools.values())

    def find_tools_for_topic(self, topic: str) -> List[ToolInfo]:
        """根据主题检索适合的工具列表。

        当前实现使用非常简单的关键词匹配策略：
            - 将 topic 转为小写
            - 若工具的 good_for 中任一关键字出现在 topic 中，即认为适配
            - 若 good_for 为空，则视为“通用工具”，始终可用
        """

        topic_lower = (topic or "").lower()
        matched: List[ToolInfo] = []

        for tool in self._tools.values():
            if not tool.good_for:
                matched.append(tool)
                continue

            for keyword in tool.good_for:
                if not keyword:
                    continue
                if keyword.lower() in topic_lower:
                    matched.append(tool)
                    break

        logger.info("按主题检索工具，topic=%r, 命中: %s", topic, matched)
        return matched

