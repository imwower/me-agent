from __future__ import annotations

from typing import List, Optional

from me_core.learning import SimpleLearner
from me_core.self_model import SimpleSelfModel
from me_core.world_model import SimpleWorldModel

from .types import IntrospectionLog


class IntrospectionGenerator:
    """基于时间线与学习统计生成简单的自省日志。"""

    def __init__(
        self,
        world: SimpleWorldModel,
        self_model: SimpleSelfModel,
        learner: SimpleLearner,
    ) -> None:
        self.world = world
        self.self_model = self_model
        self.learner = learner

    def _collect_mistakes(self) -> List[str]:
        mistakes: List[str] = []
        for name, stats in self.learner.tool_stats.items():
            if stats.call_count >= 2:
                success_rate = stats.success_count / float(stats.call_count)
                if success_rate < 0.5:
                    mistakes.append(f"工具「{name}」成功率偏低（{success_rate:.0%}），需要改进调用方式或减少使用。")
        return mistakes

    def _collect_improvements(self) -> List[str]:
        improvements: List[str] = []
        state = self.self_model.get_state()
        if len(state.seen_modalities) <= 1:
            improvements.append("尝试获取更多模态信息，丰富对世界的理解。")
        if not self.learner.intent_stats:
            improvements.append("增加意图执行回路的统计，以便更好地评估策略。")
        return improvements

    def generate(
        self,
        scenario_id: Optional[str],
        start_step: int,
        end_step: int,
    ) -> IntrospectionLog:
        events = getattr(self.world, "events_between", lambda a, b: [])(start_step, end_step)
        summary_parts: List[str] = []
        if events:
            summary_parts.append(f"本段内共有 {len(events)} 条事件。")
        summary_parts.append(self.self_model.describe_self(world_model=self.world))

        mistakes = self._collect_mistakes()
        improvements = self._collect_improvements()

        return IntrospectionLog.new(
            scenario_id=scenario_id,
            step_range=(start_step, end_step),
            summary=" ".join(summary_parts),
            mistakes=mistakes,
            improvements=improvements,
        )
