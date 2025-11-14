from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

from me_core.types import AgentEvent

from .self_state import SelfState

logger = logging.getLogger(__name__)


def _adjust_capability(
    capabilities: Dict[str, float], name: str, delta: float
) -> None:
    """根据给定增量调整某项能力，并裁剪到 [0,1]。

    若能力不存在，则从中性值 0.5 起步。
    """

    current = capabilities.get(name, 0.5)
    new_value = current + delta
    if new_value < 0.0:
        new_value = 0.0
    elif new_value > 1.0:
        new_value = 1.0
    capabilities[name] = new_value


def update_from_event(state: SelfState, event: AgentEvent) -> SelfState:
    """根据单个 AgentEvent 更新 SelfState。

    约定的事件结构示例（payload）：
        {
            "kind": "task",           # 或 "error" / "info" 等
            "task_type": "summarize", # 任务类别/能力名称
            "success": True,          # 是否成功
            "topic": "自我模型设计",    # 关联主题
            "error": "错误原因...",     # 若失败可携带
        }

    简单规则：
        - task + success=True:
            - 对应能力熟练度略微提升（+0.05）
            - topic 若存在，则加入 focus_topics
            - recent_activities 追加成功描述
        - task + success=False:
            - 对应能力熟练度略微降低（-0.05）
            - 若有 error，追加到 limitations（去重）
        - event_type 为 "error" 的事件：
            - 将描述性信息追加到 limitations
    """

    new_state = replace(state)
    payload = event.payload or {}

    logger.info("根据事件更新自我状态，原状态: %s, 事件: %s", state, event)

    kind = payload.get("kind") or event.event_type
    task_type = payload.get("task_type")
    success = payload.get("success")
    topic = payload.get("topic")
    error_msg = payload.get("error")

    # 根据任务成功/失败更新能力与主题
    if kind == "task" and isinstance(task_type, str):
        if success is True:
            _adjust_capability(new_state.capabilities, task_type, 0.05)
            if isinstance(topic, str) and topic:
                if topic not in new_state.focus_topics:
                    new_state.focus_topics.append(topic)
            new_state.add_activity(f"成功完成任务: {task_type}")
        elif success is False:
            _adjust_capability(new_state.capabilities, task_type, -0.05)
            if isinstance(error_msg, str) and error_msg:
                if error_msg not in new_state.limitations:
                    new_state.limitations.append(error_msg)
            new_state.add_activity(f"任务失败: {task_type}")

    # 感知事件：记录最近感知到的信息，便于自我总结中体现“最近在看什么”
    if kind == "perception":
        text_snippet = ""
        raw = payload.get("raw") if isinstance(payload, dict) else None
        if isinstance(raw, dict):
            text_value = raw.get("text")
            if isinstance(text_value, str) and text_value:
                # 只截取前若干字符，避免活动描述过长
                text_snippet = text_value[:30]

        if text_snippet:
            activity_desc = f"感知到新的输入片段：{text_snippet}"
        else:
            activity_desc = "感知到新的多模态输入"

        new_state.add_activity(activity_desc)

    # 独立的错误事件，也会推动自我局限的更新
    if event.event_type == "error":
        desc = error_msg or payload.get("message")
        if isinstance(desc, str) and desc:
            if desc not in new_state.limitations:
                new_state.limitations.append(desc)
            new_state.add_activity(f"遇到错误: {desc}")

    logger.info("事件更新后的自我状态: %s", new_state)
    return new_state


def _collect_task_stats(
    history: Iterable[AgentEvent],
) -> Dict[str, Tuple[int, int]]:
    """从事件历史中收集每种任务的成功/失败次数。

    返回：
        capability_name -> (success_count, failure_count)
    """

    stats: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))

    for event in history:
        payload = event.payload or {}
        kind = payload.get("kind") or event.event_type
        if kind != "task":
            continue

        task_type = payload.get("task_type")
        success = payload.get("success")
        if not isinstance(task_type, str):
            continue

        success_count, failure_count = stats[task_type]
        if success is True:
            success_count += 1
        elif success is False:
            failure_count += 1
        stats[task_type] = (success_count, failure_count)

    return stats


def aggregate_stats(state: SelfState, history: List[AgentEvent]) -> SelfState:
    """根据一段历史事件聚合统计结果，重新评估能力与局限。

    简单策略：
        - 对每类任务计算成功率 p = success / (success + failure)
        - 将能力值设置为 0.3 + 0.7 * p，使得：
            - 即便成功率为 0，能力也不会被认为完全不存在（保持 0.3 的基础）
            - 成功率越高，对应能力值越接近 1.0
        - 若某能力失败次数远大于成功次数，则将一条描述性局限加入 limitations。
    """

    new_state = replace(state)
    # 在聚合前记录一份旧的能力快照，用于后续计算“能力变化趋势”
    previous_capabilities = dict(state.capabilities)
    stats = _collect_task_stats(history)

    logger.info("根据历史事件聚合更新自我状态，原状态: %s, 统计: %s", state, stats)

    for capability, (success_count, failure_count) in stats.items():
        total = success_count + failure_count
        if total == 0:
            continue
        success_ratio = success_count / total
        # 使用简单线性插值将成功率映射到 [0.3, 1.0]
        value = 0.3 + 0.7 * success_ratio
        if value < 0.0:
            value = 0.0
        elif value > 1.0:
            value = 1.0
        new_state.capabilities[capability] = value

        # 若失败明显多于成功，记录局限
        if failure_count >= success_count * 2 and capability not in new_state.limitations:
            new_state.limitations.append(f"在任务 '{capability}' 上容易出错")

    # 计算能力变化趋势：仅记录变化幅度超过一定阈值的能力
    trend: Dict[str, float] = {}
    for capability, new_value in new_state.capabilities.items():
        old_value = previous_capabilities.get(capability, 0.5)
        delta = new_value - old_value
        if abs(delta) < 0.05:
            continue
        trend[capability] = delta

    new_state.capability_trend = trend

    logger.info("聚合历史后的自我状态: %s, 能力变化趋势: %s", new_state, trend)
    return new_state
