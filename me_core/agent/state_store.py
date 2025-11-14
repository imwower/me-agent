from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

from me_core.drives.drive_vector import DriveVector
from me_core.self_model.self_state import SelfState

logger = logging.getLogger(__name__)


@dataclass
class StateStore:
    """使用本地 JSON 文件持久化智能体核心状态。

    当前存储内容：
        - SelfState：自我模型状态
        - DriveVector：内在驱动力
        - event_summaries：少量历史事件的文本摘要（仅用于简单回顾）
    """

    path: Path = field(default_factory=lambda: Path("agent_state.json"))
    self_state: SelfState = field(default_factory=SelfState)
    drives: DriveVector = field(default_factory=DriveVector)
    event_summaries: List[str] = field(default_factory=list)

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

        self.self_state = SelfState.from_dict(self_state_data)
        self.drives = DriveVector.from_dict(drives_data)
        self.event_summaries = list(events)

    def save_state(self) -> None:
        """将当前状态保存到 JSON 文件。"""

        data = {
            "self_state": self.self_state.to_dict(),
            "drives": self.drives.as_dict(),
            "event_summaries": list(self.event_summaries),
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

