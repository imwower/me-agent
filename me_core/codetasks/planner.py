from __future__ import annotations

import uuid
from typing import List

from me_core.introspection import IntrospectionLog
from me_core.teachers.types import TeacherOutput
from me_core.tasks.types import TaskResult

from .types import CodeTask


class CodeTaskPlanner:
    """根据内省和 Teacher 建议生成代码任务（占位实现）。"""

    def plan_tasks(
        self,
        repo_id: str,
        introspection: IntrospectionLog | None,
        teacher_outputs: List[TeacherOutput],
        task_result: TaskResult | None,
    ) -> List[CodeTask]:
        tasks: List[CodeTask] = []
        description_bits: List[str] = []
        if introspection:
            description_bits.append(introspection.summary)
            description_bits.extend(introspection.mistakes)
            description_bits.extend(introspection.improvements)
        for t in teacher_outputs:
            if t.advice_text:
                description_bits.append(t.advice_text)
        title = "根据内省与建议改进代码"
        files = []
        if task_result and task_result.details:
            for step in task_result.details.get("steps", []):
                path = step.get("image_path")
                if path:
                    files.append(path)
        constraints = ["仅使用标准库 unless 扩展", "保持现有接口兼容"]
        if introspection and introspection.summary and "脑" in introspection.summary:
            constraints.append("针对脑结构改动需保持训练脚本可运行，中文注释说明改动原因")
        tasks.append(
            CodeTask(
                id=str(uuid.uuid4()),
                repo_id=repo_id,
                title=title,
                description="\n".join(description_bits) or "改进代码质量和测试稳定性。",
                files_to_read=list(set(files)),
                files_to_edit=[],
                constraints=constraints,
                acceptance_criteria=["单测通过", "满足 Teacher/内省建议"],
            )
        )
        return tasks
