from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from me_core.memory import EpisodicMemory, SemanticMemory
from me_core.world_model import SimpleWorldModel
from .self_state import SelfState


def generate_long_term_report(
    self_state: SelfState,
    world: SimpleWorldModel,
    episodic: EpisodicMemory,
    semantic: SemanticMemory,
    horizon_steps: int = 1000,
) -> str:
    """
    生成长期自我总结报告。

    包含：
    - 最近任务/事件类型
    - 工具使用统计
    - 脑态摘要（若有 BrainSnapshot）
    - 记忆中出现的概念/情节数量
    """

    lines: List[str] = []
    lines.append("# 长期自我总结")
    lines.append(f"- 时间: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- 最近自我模式: {self_state.last_brain_mode} (置信 {self_state.last_brain_confidence:.2f})")

    # 世界事件与工具使用
    summary = world.summarize()
    events_summary = summary.get("events", {})
    lines.append("## 事件回顾")
    for k, v in events_summary.items():
        lines.append(f"- 事件类型 {k}: {v}")

    tools = summary.get("tools", {})
    if tools:
        lines.append("## 工具使用")
        for name, stats in tools.items():
            lines.append(
                f"- {name}: 成功 {stats.get('success',0)} 次 / 失败 {stats.get('failure',0)} 次，成功率 {stats.get('success_rate')}"
            )

    # 脑态
    brain = summary.get("brain", {})
    lines.append("## 脑状态")
    lines.append(f"- 最近脑模式: {brain.get('last_mode')}")
    if brain.get("last_global_metrics"):
        metrics_str = "; ".join(f"{k}={v}" for k, v in brain.get("last_global_metrics", {}).items())
        lines.append(f"- 全局指标: {metrics_str}")

    # 记忆
    eps = episodic.recent_episodes()
    lines.append(f"## 记忆摘要 (最近 {len(eps)} 条情节)")
    for ep in eps[-5:]:
        lines.append(f"- [{ep.id}] step {ep.start_step}-{ep.end_step}: {ep.summary or '无摘要'}")

    concept_mems = semantic.all_memories()
    lines.append(f"## 概念记忆 ({len(concept_mems)} 个)")
    for cm in concept_mems[:5]:
        lines.append(f"- {cm.name}: {cm.description}")

    lines.append("## 自我能力与需求")
    lines.append(f"- 能力标签: {', '.join(sorted(self_state.capability_tags)) or '暂无'}")
    lines.append(f"- 关注主题: {', '.join(self_state.focus_topics) or '暂无'}")
    lines.append(f"- 需求: {', '.join(self_state.needs) or '暂无'}")

    return "\n".join(lines)
