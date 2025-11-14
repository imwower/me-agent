from __future__ import annotations

import logging
from typing import Dict, List

from .self_state import SelfState

logger = logging.getLogger(__name__)


def _format_capabilities(capabilities: Dict[str, float]) -> str:
    """将能力字典转换为可读的中文描述。"""

    if not capabilities:
        return "我目前还没有明确标记的能力，但正在逐步学习。"

    # 按熟练度从高到低排序，重点展示前几项
    items = sorted(
        capabilities.items(), key=lambda x: x[1], reverse=True
    )
    lines: List[str] = []
    for name, level in items[:5]:
        # 使用简单的区间来口语化熟练度
        if level >= 0.8:
            degree = "非常擅长"
        elif level >= 0.6:
            degree = "比较擅长"
        elif level >= 0.4:
            degree = "有一定经验"
        else:
            degree = "还在练习"
        lines.append(f"{degree}「{name}」")

    return "；".join(lines)


def _format_needs(needs: List[str]) -> str:
    """将当前需要列表转换为可读的中文描述。"""

    if not needs:
        return "目前没有特别紧急的需要，如果你有任务或反馈，我会很乐意参与。"

    return "；".join(needs)


def _format_recent_activities(activities: List[str]) -> str:
    """将最近活动列表转为简短可读的中文描述。

    只展示最新的若干条，避免自我总结过长。
    """

    if not activities:
        return ""

    # 仅展示最近 3 条活动，从旧到新排列，让读者看到“最近在做什么”
    recent = activities[-3:]
    text = "；".join(recent)
    result = f"最近我做过的一些事情包括：{text}。"
    logger.info("格式化最近活动: %s -> %s", recent, result)
    return result


def _format_recent_perceptions(activities: List[str]) -> str:
    """从最近活动中提取感知相关的信息，生成专门的描述。

    约定：
        - 感知事件由 self_updater 使用前缀 "感知到新的输入片段：" 记录；
        - 这里只取最近几条感知活动，避免重复与过长。
    """

    prefix = "感知到新的输入片段："
    perception_snippets: List[str] = []

    for item in activities:
        if item.startswith(prefix):
            snippet = item[len(prefix) :].strip()
            if snippet:
                perception_snippets.append(snippet)

    if not perception_snippets:
        return ""

    # 只展示最近 2 条感知片段，让总结更简洁
    recent = perception_snippets[-2:]
    text = "；".join(recent)
    result = f"在感知层面，我最近注意到：{text}。"
    logger.info("格式化最近感知片段: %s -> %s", recent, result)
    return result


def summarize_self(state: SelfState) -> Dict[str, str]:
    """根据 SelfState 生成一段简短的自我总结。

    返回字典包含三个键：
        - who_am_i: 一句简短自述“我是谁”
        - what_can_i_do: 几句说明“我能做什么、我擅长什么”
        - what_do_i_need: 几句说明“我目前缺什么、需要什么帮助”
    """

    logger.info("生成自我总结，当前自我状态: %s", state)

    who_am_i = f"我是 {state.identity}。"

    capability_text = _format_capabilities(state.capabilities)
    # 将关注主题与最近活动也融入“我能做什么”的描述，形成更具时间感的自述
    if state.focus_topics:
        topics = "、".join(state.focus_topics[:5])
        capability_text += f"；最近我特别关注的方向包括：{topics}。"

    recent_text = _format_recent_activities(state.recent_activities)
    if recent_text:
        capability_text += f" {recent_text}"

    perception_text = _format_recent_perceptions(state.recent_activities)
    if perception_text:
        capability_text += f" {perception_text}"

    needs_text = _format_needs(state.needs)
    if state.limitations:
        # 仅展示部分局限，避免过长
        lim = "；".join(state.limitations[:3])
        needs_text += f" 同时，我也清楚自己的局限，例如：{lim}。"

    summary = {
        "who_am_i": who_am_i,
        "what_can_i_do": capability_text,
        "what_do_i_need": needs_text,
    }

    logger.info("自我总结结果: %s", summary)
    return summary
