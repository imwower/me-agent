from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from me_core.types import ToolResult as CoreToolResult

from .registry import ToolInfo

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolResult:
    """工具执行结果的简单结构。

    字段：
        tool_name: 工具名称
        success: 是否执行成功（对 stub 工具而言通常为 True）
        summary: 面向人的简要描述
        details: 结构化细节，供后续学习模块使用
    """

    tool_name: str
    success: bool
    summary: str
    details: Dict[str, Any]


def to_core_tool_result(stub_result: ToolResult, call_id: str) -> CoreToolResult:
    """将桩实现的 ToolResult 转换为核心 ToolResult 结构。

    这样在需要统一日志/事件格式时，可以方便地复用同一套类型定义。
    """

    logger.info(
        "转换桩工具结果为核心 ToolResult: tool_name=%s, call_id=%s",
        stub_result.tool_name,
        call_id,
    )

    output: Dict[str, Any] = {
        "tool_name": stub_result.tool_name,
        "summary": stub_result.summary,
        "details": stub_result.details,
    }

    return CoreToolResult(
        call_id=call_id,
        success=stub_result.success,
        output=output,
        error=None,
        meta=None,
    )


class ToolExecutorStub:
    """工具执行器桩实现。

    用于在没有真实外部系统时，模拟工具调用返回结果。
    当前内置两个虚拟工具：
        - "search_papers": 返回伪造的论文摘要与关键词
        - "run_simulation": 返回伪造的实验结果与性能指标
    """

    def execute(self, tool: ToolInfo, args: Dict[str, Any]) -> ToolResult:
        """执行指定工具，返回伪造的结果。

        参数：
            tool: 要执行的工具信息
            args: 调用参数，例如 {"topic": "...", "context": {...}}
        """

        topic = str(args.get("topic") or "未知主题")
        logger.info("执行工具: %s, topic=%s, args=%s", tool.name, topic, args)

        if tool.name == "search_papers":
            # 伪造论文检索结果
            papers = [
                {
                    "title": f"关于「{topic}」的理论分析",
                    "summary": f"提出了一种针对「{topic}」的基础模型与训练方法。",
                },
                {
                    "title": f"在实际系统中应用「{topic}」的经验报告",
                    "summary": f"总结了在真实场景中部署「{topic}」时的若干挑战与经验。",
                },
            ]
            details: Dict[str, Any] = {
                "topic": topic,
                "papers": papers,
                "keywords": [topic, "paper", "study"],
            }
            summary = f"针对主题「{topic}」检索到 {len(papers)} 篇相关论文。"
            result = ToolResult(
                tool_name=tool.name,
                success=True,
                summary=summary,
                details=details,
            )
        elif tool.name == "run_simulation":
            # 伪造实验模拟结果
            improved = True
            details = {
                "topic": topic,
                "improved": improved,
                "metrics": {
                    "baseline_score": 0.72,
                    "new_score": 0.81,
                },
            }
            summary = (
                f"针对主题「{topic}」完成了一次模拟实验，"
                "观察到相对于基线有一定性能提升。"
            )
            result = ToolResult(
                tool_name=tool.name,
                success=True,
                summary=summary,
                details=details,
            )
        elif tool.name == "codex":
            # 伪造 Codex 风格的“智能回答”
            prompt = str(args.get("prompt") or topic)
            simulated_answer = (
                f"（Codex 桩实现）围绕「{topic}」给出了一段解释性回答，"
                f"大致基于提示：{prompt[:80]}..."
            )
            details = {
                "topic": topic,
                "prompt": prompt,
                "simulated_answer": simulated_answer,
            }
            summary = f"Codex 针对主题「{topic}」返回了一段说明性内容（桩实现）。"
            result = ToolResult(
                tool_name=tool.name,
                success=True,
                summary=summary,
                details=details,
            )
        else:
            # 未知工具采用通用兜底逻辑
            details = {
                "topic": topic,
                "message": "该工具为桩实现，仅返回占位结果。",
            }
            summary = f"执行工具「{tool.name}」完成（桩实现），未进行真实外部操作。"
            result = ToolResult(
                tool_name=tool.name,
                success=True,
                summary=summary,
                details=details,
            )

        logger.info("工具执行结果: %s", result)
        return result
