from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(slots=True)
class DriveVector:
    """表示智能体的一组内在驱动力参数。

    所有字段取值范围约定为 [0.0, 1.0]：
    - chat_level: 话痨度，越高越倾向于主动说话/展开解释
    - curiosity_level: 好奇心，越高越愿意去探索不确定/新事物
    - exploration_level: 探索欲，越高越倾向于尝试新策略、新工具
    - learning_intensity: 学习强度，越高越频繁地更新知识、做实验
    - social_need: 社交需求，越高越希望与用户互动
    - data_need: 数据需求，越高越渴望获取新的数据/样本
    """

    chat_level: float = 0.5
    curiosity_level: float = 0.5
    exploration_level: float = 0.5
    learning_intensity: float = 0.5
    social_need: float = 0.5
    data_need: float = 0.5

    def clamp(self) -> None:
        """将所有驱动力值裁剪到 [0.0, 1.0] 范围内。

        该方法会直接原地修改实例自身。
        """

        # 按字段逐个裁剪，确保不会出现越界值
        for field_name in (
            "chat_level",
            "curiosity_level",
            "exploration_level",
            "learning_intensity",
            "social_need",
            "data_need",
        ):
            value = getattr(self, field_name)
            if value < 0.0:
                value = 0.0
            elif value > 1.0:
                value = 1.0
            setattr(self, field_name, value)

    def as_dict(self) -> Dict[str, float]:
        """将当前驱动力向量序列化为字典。

        返回的字典使用简单的标量 float，方便持久化或日志记录。
        """

        return {
            "chat_level": self.chat_level,
            "curiosity_level": self.curiosity_level,
            "exploration_level": self.exploration_level,
            "learning_intensity": self.learning_intensity,
            "social_need": self.social_need,
            "data_need": self.data_need,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> "DriveVector":
        """从字典恢复一个 DriveVector 实例。

        对于缺失的字段，使用类的默认值（0.5）。
        输入中若有额外字段会被忽略。
        """

        # 读取各字段，如字典中不存在则退回到默认值
        return cls(
            chat_level=float(data.get("chat_level", 0.5)),
            curiosity_level=float(data.get("curiosity_level", 0.5)),
            exploration_level=float(data.get("exploration_level", 0.5)),
            learning_intensity=float(data.get("learning_intensity", 0.5)),
            social_need=float(data.get("social_need", 0.5)),
            data_need=float(data.get("data_need", 0.5)),
        )

