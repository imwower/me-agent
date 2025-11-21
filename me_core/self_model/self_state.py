from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional, Set


@dataclass(slots=True)
class SelfState:
    """描述智能体对“自我”的当前认识。

    字段说明：
        identity: 对自身角色的简短描述，例如："一个专注于帮助用户思考的 AI 助手"
        capabilities: 能力名称 -> 熟练度 [0,1]，例如 {"summarize": 0.8}
        focus_topics: 最近关注的主题列表，例如 ["自我模型", "驱动力设计"]
        limitations: 当前自认为的主要局限（文本描述），例如 ["不了解最新的现实世界数据"]
        recent_activities: 最近若干条活动摘要，按时间顺序排列，越新的在列表尾部
        needs: 当前主要“需要”，例如 ["需要更多用户反馈", "需要更多实验数据"]
    """

    identity: str = "一个正在学习如何更好理解自我的智能体"
    capabilities: Dict[str, float] = field(default_factory=dict)
    focus_topics: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    recent_activities: List[str] = field(default_factory=list)
    needs: List[str] = field(default_factory=list)
    # 记录最近一次聚合统计后，各能力相对于之前的变化趋势：能力名称 -> Δ值
    capability_trend: Dict[str, float] = field(default_factory=dict)
    # 与多模态/概念空间相关的自我认知字段
    self_concept_id: Optional[str] = None
    capability_tags: Set[str] = field(default_factory=set)
    modalities_seen: Set[str] = field(default_factory=set)
    seen_modalities: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        # 确保新老字段保持同步，避免两套记录分叉
        combined = set(self.modalities_seen) | set(self.seen_modalities)
        self.modalities_seen = combined
        self.seen_modalities = combined

    def add_activity(self, desc: str, max_len: int = 20) -> None:
        """追加一条活动摘要，并在超出长度时裁剪旧的记录。

        参数：
            desc: 活动的简短描述，例如 "完成了一次文本总结任务"
            max_len: recent_activities 的最大长度，超过时从最旧的开始丢弃
        """

        if not desc:
            return
        self.recent_activities.append(desc)
        # 只保留最后 max_len 条记录
        if len(self.recent_activities) > max_len:
            overflow = len(self.recent_activities) - max_len
            del self.recent_activities[0:overflow]

    def to_dict(self) -> Dict[str, Any]:
        """将自我状态转换为字典，方便持久化或序列化。"""

        # 默认 asdict 会保留 set，JSON 序列化会失败，这里手工转换。
        data = asdict(self)
        # capability_tags / modalities_seen 为 set，需转为列表便于 json.dump
        data["capability_tags"] = list(self.capability_tags)
        data["modalities_seen"] = list(self.modalities_seen)
        data["seen_modalities"] = list(self.seen_modalities)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelfState":
        """从字典构造 SelfState 实例。

        对缺失字段使用默认值，对于未知字段会忽略。
        """

        return cls(
            identity=str(data.get("identity", "一个正在学习如何更好理解自我的智能体")),
            capabilities=dict(data.get("capabilities", {})) or {},
            focus_topics=list(data.get("focus_topics", [])) or [],
            limitations=list(data.get("limitations", [])) or [],
            recent_activities=list(data.get("recent_activities", [])) or [],
            needs=list(data.get("needs", [])) or [],
            capability_trend=dict(data.get("capability_trend", {})) or {},
            self_concept_id=data.get("self_concept_id"),
            capability_tags=set(data.get("capability_tags", []) or []),
            modalities_seen=set(data.get("modalities_seen", []) or []),
            seen_modalities=set(
                data.get("seen_modalities", []) or data.get("modalities_seen", []) or []
            ),
        )


def default_self_state() -> SelfState:
    """构造一个带有合理默认值的 SelfState 实例。

    设计意图：
        - 作为 AgentState 初始化时的“自我模型基线”；
        - 便于在不同场景中快速获得一致的起点，而不是每次手写字段。
    """

    return SelfState()
