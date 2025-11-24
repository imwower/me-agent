from __future__ import annotations

from typing import Dict, List

from .types import CodeTask


class PromptGenerator:
    """将 CodeTask + 文件内容拼装成适合 Code-LLM 的提示词。"""

    def generate(self, task: CodeTask, file_contents: Dict[str, str]) -> str:
        lines: List[str] = []
        lines.append(f"你现在是本项目的协同开发 AI。任务编号：{task.id}，仓库：{task.repo_id}")
        lines.append("\n目标：")
        lines.append(task.description)
        if task.constraints:
            lines.append("\n约束：")
            for c in task.constraints:
                lines.append(f"- {c}")
        if task.acceptance_criteria:
            lines.append("\n验收标准：")
            for a in task.acceptance_criteria:
                lines.append(f"- {a}")
        if task.files_to_read:
            lines.append("\n相关文件内容：")
            for path in task.files_to_read:
                content = file_contents.get(path, "")
                snippet = content[:2000]
                lines.append(f"- {path}:\n```python\n{snippet}\n```")
        lines.append("\n请给出修改方案，并直接输出建议的代码 diff 或更新后的文件内容。")
        return "\n".join(lines)


__all__ = ["PromptGenerator"]
