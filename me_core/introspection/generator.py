from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from me_core.learning import SimpleLearner
from me_core.self_model import SimpleSelfModel
from me_core.world_model import SimpleWorldModel

if TYPE_CHECKING:  # 避免循环导入
    from me_core.tasks.experiment_types import ExperimentResult

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
        test_failures: Optional[List[str]] = None,
        notes: Optional[str] = None,
        experiment_results: Optional["List[ExperimentResult]"] = None,
    ) -> IntrospectionLog:
        events = getattr(self.world, "events_between", lambda a, b: [])(start_step, end_step)
        summary_parts: List[str] = []
        if events:
            summary_parts.append(f"本段内共有 {len(events)} 条事件。")
        summary_parts.append(self.self_model.describe_self(world_model=self.world))
        if notes:
            summary_parts.append(notes)

        mistakes = self._collect_mistakes()
        if test_failures:
            mistakes.append(f"以下测试失败：{', '.join(test_failures)}")

        improvements = self._collect_improvements()
        is_code_scene = (scenario_id and any(key in scenario_id for key in ("dev", "code"))) or (
            notes and any(key in notes for key in ("dev", "code"))
        )
        if is_code_scene:
            improvements.append("针对代码场景，建议补充单元测试并复查最近的修改。")
        if experiment_results:
            for res in experiment_results:
                if res.metrics:
                    summary_parts.append(f"实验 {res.step.kind} 指标：{res.metrics}")
                if res.returncode != 0:
                    mistakes.append(f"实验步骤 {res.step.kind} 退出码 {res.returncode}")

        return IntrospectionLog.new(
            scenario_id=scenario_id,
            step_range=(start_step, end_step),
            summary=" ".join(summary_parts),
            mistakes=mistakes,
            improvements=improvements,
        )
