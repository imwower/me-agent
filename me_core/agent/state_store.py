from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from me_core.drives.drive_vector import DriveVector
from me_core.self_model.self_state import SelfState
from me_core.types import AgentEvent

logger = logging.getLogger(__name__)


@dataclass
class StateStore:
    """使用本地 JSON 文件持久化智能体核心状态。

    当前存储内容：
        - SelfState：自我模型状态
        - DriveVector：内在驱动力
        - event_summaries：少量历史事件的文本摘要（仅用于简单回顾）
        - events：最近若干条结构化 AgentEvent（用于统计与自我更新）
        - knowledge_base：学习模块内部维护的简单知识条目列表
    """

    path: Path = field(default_factory=lambda: Path("agent_state.json"))
    self_state: SelfState = field(default_factory=SelfState)
    drives: DriveVector = field(default_factory=DriveVector)
    event_summaries: List[str] = field(default_factory=list)
    events: List[AgentEvent] = field(default_factory=list)
    knowledge_base: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """初始化时尝试从磁盘加载已有状态。"""

        self.load_state()

    def load_state(self) -> None:
        """从 JSON 文件加载状态，若不存在则使用默认值。"""

        if not self.path.exists():
            logger.info("状态文件不存在，将使用默认状态: %s", self.path)
            return

        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取状态文件失败，将重置为默认状态: %s", exc)
            return

        logger.info("从状态文件加载状态: %s", self.path)

        self_state_data = data.get("self_state") or {}
        drives_data = data.get("drives") or {}
        events = data.get("event_summaries") or []
        raw_events = data.get("events") or []
        kb = data.get("knowledge_base") or []

        self.self_state = SelfState.from_dict(self_state_data)
        self.drives = DriveVector.from_dict(drives_data)
        self.event_summaries = list(events)

        # 将历史事件从字典恢复为 AgentEvent，忽略异常项
        restored_events: List[AgentEvent] = []
        for item in raw_events:
            if isinstance(item, dict):
                try:
                    restored_events.append(AgentEvent.from_dict(item))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("反序列化事件失败，将跳过该条记录: %s", exc)
        self.events = restored_events
        # 知识库直接以列表形式保存/恢复，具体结构由 LearningManager 约定
        if isinstance(kb, list):
            self.knowledge_base = list(kb)
        else:
            self.knowledge_base = []

    def save_state(self) -> None:
        """将当前状态保存到 JSON 文件。"""

        data = {
            "self_state": self.self_state.to_dict(),
            "drives": self.drives.as_dict(),
            "event_summaries": list(self.event_summaries),
            "events": [e.to_dict() for e in self.events],
            "knowledge_base": list(self.knowledge_base),
        }

        try:
            with self.path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.error("写入状态文件失败: %s", exc)
        else:
            logger.info("状态已保存到: %s", self.path)

    def get_self_state(self) -> SelfState:
        """获取当前自我状态。"""

        return self.self_state

    def set_self_state(self, state: SelfState) -> None:
        """更新当前自我状态。"""

        logger.info("更新自我状态: %s", state)
        self.self_state = state

    def get_drives(self) -> DriveVector:
        """获取当前驱动力向量。"""

        return self.drives

    def set_drives(self, drives: DriveVector) -> None:
        """更新当前驱动力向量。"""

        logger.info("更新驱动力向量: %s", drives)
        self.drives = drives

    def add_event_summary(self, summary: str, max_len: int = 50) -> None:
        """追加一条事件摘要，并限制总条数。"""

        if not summary:
            return
        self.event_summaries.append(summary)
        if len(self.event_summaries) > max_len:
            overflow = len(self.event_summaries) - max_len
            del self.event_summaries[0:overflow]

    def set_knowledge_base(self, knowledge_base: List[Dict[str, Any]]) -> None:
        """设置当前的知识库内容。"""

        logger.info("更新知识库，当前条目数: %d", len(knowledge_base))
        self.knowledge_base = list(knowledge_base)

    def get_knowledge_base(self) -> List[Dict[str, Any]]:
        """获取当前知识库内容。"""

        return list(self.knowledge_base)

    def append_events(self, new_events: List[AgentEvent], max_len: int = 100) -> None:
        """追加结构化事件，并限制总条数。

        这为后续的统计与聚合（例如 aggregate_stats）提供原始数据来源。
        """

        if not new_events:
            return

        logger.info("追加 %d 条事件到状态存储。", len(new_events))
        self.events.extend(new_events)
        if len(self.events) > max_len:
            overflow = len(self.events) - max_len
            del self.events[0:overflow]

    def get_events(self, limit: int | None = None) -> List[AgentEvent]:
        """获取最近若干条事件。

        参数：
            limit: 若为 None，则返回全部事件；否则返回最近 limit 条。
        """

        if limit is None or limit >= len(self.events):
            return list(self.events)
        return self.events[-limit:]
