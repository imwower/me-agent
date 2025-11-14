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

        一个简单的启发式公式：
            learning_desire = uncertainty * curiosity * learning_intensity
        并将结果裁剪到 [0,1]。
        """

        raw = (
            float(uncertainty)
            * float(drives.curiosity_level)
            * float(drives.learning_intensity)
        )
        if raw < 0.0:
            raw = 0.0
        elif raw > 1.0:
            raw = 1.0

        logger.info(
            "计算学习意愿: uncertainty=%.3f, curiosity=%.3f, learning_intensity=%.3f, result=%.3f",
            uncertainty,
            drives.curiosity_level,
            drives.learning_intensity,
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
        candidate_tools: List[ToolInfo] = self.registry.find_tools_for_topic(
            topic
        )

        if not candidate_tools:
            logger.info("未找到适合主题 %r 的工具，跳过学习。", topic)
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

