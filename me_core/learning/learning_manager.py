from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from me_core.drives.drive_vector import DriveVector
from me_core.tools.executor_stub import ToolExecutorStub, ToolResult
from me_core.tools.registry import ToolInfo, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class LearningManager:
    """学习管理器。

    职责：
        - 根据“不确定性 + 内在驱动力”计算当前的学习意愿（learning_desire）。
        - 在学习意愿足够高时，选择合适的工具发起“学习型”调用。
        - 将有价值的结果写入简单的内存知识库（knowledge_base）。
    """

    registry: ToolRegistry
    knowledge_base: List[Dict[str, Any]] = field(default_factory=list)
    executor: ToolExecutorStub = field(default_factory=ToolExecutorStub)

    def compute_learning_desire(
        self,
        uncertainty: float,
        drives: DriveVector,
    ) -> float:
        """根据不确定性与驱动力计算当前的学习意愿。

        一个简单的启发式公式（保持可解释性为主）：
            base = uncertainty * curiosity * learning_intensity
            调整因子 adjustment ≈ (exploration_level + data_need) / 2
        在保证结果位于 [0,1] 的前提下，让“更愿意探索且更缺数据”的状态
        有更高概率触发主动学习行为。
        并将结果裁剪到 [0,1]。
        """

        uncertainty_f = float(uncertainty)
        curiosity_f = float(drives.curiosity_level)
        learning_intensity_f = float(drives.learning_intensity)
        exploration_f = float(drives.exploration_level)
        data_need_f = float(drives.data_need)

        base = uncertainty_f * curiosity_f * learning_intensity_f
        # 将探索欲与数据需求归一到 [0,1] 后，取平均作为调整因子
        adjustment = (exploration_f + data_need_f) / 2.0
        raw = base * (0.5 + 0.5 * adjustment)

        if raw < 0.0:
            raw = 0.0
        elif raw > 1.0:
            raw = 1.0

        logger.info(
            (
                "计算学习意愿: uncertainty=%.3f, curiosity=%.3f, "
                "learning_intensity=%.3f, exploration=%.3f, data_need=%.3f, "
                "base=%.3f, adjustment=%.3f, result=%.3f"
            ),
            uncertainty_f,
            curiosity_f,
            learning_intensity_f,
            exploration_f,
            data_need_f,
            base,
            adjustment,
            raw,
        )
        return raw

    def plan_learning_topic(self, context: Dict[str, Any]) -> str:
        """从上下文中提取当前学习主题。

        优先级：
            1. context["topic"]
            2. context["focus_topic"]
            3. context["question"] / context["description"]
            4. 默认使用一个泛化的主题说明。
        """

        topic = (
            context.get("topic")
            or context.get("focus_topic")
            or context.get("question")
            or context.get("description")
        )

        if not topic:
            topic = "通用学习与自我改进"

        topic_str = str(topic)
        logger.info("规划学习主题: %s", topic_str)
        return topic_str

    def select_tools_for_topic(self, topic: str) -> List[ToolInfo]:
        """根据主题选择本轮要使用的工具列表。

        策略：
            1. 先通过注册表的关键词匹配找到“最相关”的工具；
            2. 若没有任何匹配结果，则退回到“使用当前所有已注册的工具”，
               避免因为主题表述方式不同而完全放弃学习机会。
        """

        matched = self.registry.find_tools_for_topic(topic)
        if matched:
            logger.info(
                "按主题匹配到的学习工具: topic=%r, tools=%s",
                topic,
                matched,
            )
            return matched

        all_tools = self.registry.list_tools()
        if not all_tools:
            logger.info(
                "当前注册表中没有任何工具，无法执行学习。topic=%r",
                topic,
            )
            return []

        logger.info(
            "按主题未匹配到工具，将退回使用全部工具: topic=%r, tools=%s",
            topic,
            all_tools,
        )
        return all_tools

    def maybe_learn(
        self,
        uncertainty: float,
        drives: DriveVector,
        context: Dict[str, Any],
        threshold: float = 0.2,
    ) -> List[ToolResult]:
        """在学习意愿足够高时主动调用工具“学习”。

        过程：
            1. 根据 uncertainty + drives 计算 learning_desire。
            2. 若低于阈值，则直接返回空列表。
            3. 选择当前学习主题 topic。
            4. 从 registry 中检索适合该主题的工具列表。
            5. 调用 executor.execute 依次执行工具。
            6. 对于成功的结果，将简要信息写入 knowledge_base。
        """

        desire = self.compute_learning_desire(uncertainty, drives)
        if desire < threshold:
            logger.info(
                "学习意愿不足，跳过主动学习。desire=%.3f, threshold=%.3f",
                desire,
                threshold,
            )
            return []

        topic = self.plan_learning_topic(context)
        candidate_tools: List[ToolInfo] = self.select_tools_for_topic(topic)

        if not candidate_tools:
            # select_tools_for_topic 已经对“完全无工具”的情况做过日志记录
            return []

        results: List[ToolResult] = []
        logger.info(
            "开始主动学习: topic=%r, learning_desire=%.3f, 工具数=%d",
            topic,
            desire,
            len(candidate_tools),
        )

        for tool in candidate_tools:
            result = self.executor.execute(
                tool,
                {
                    "topic": topic,
                    "context": context,
                    "learning_desire": desire,
                },
            )
            results.append(result)

            if result.success:
                entry = {
                    "tool_name": result.tool_name,
                    "topic": topic,
                    "summary": result.summary,
                    "details": result.details,
                }
                self.knowledge_base.append(entry)
                logger.info("将学习结果写入知识库: %s", entry)

        logger.info(
            "本轮学习结束，共调用工具 %d 次，成功结果 %d 条，知识库大小=%d",
            len(results),
            sum(1 for r in results if r.success),
            len(self.knowledge_base),
        )
        return results
