from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PolicyStats:
    attempts: int = 0
    successes: int = 0
    total_reward: float = 0.0


class PolicyLearner:
    """
    基于 reward 的简易策略学习器：
    - 针对 AgentPolicy 中的若干可调参数（例如 curiosity/工具选择等），
      根据实验/场景结果的 reward 做微调。
    """

    def __init__(self) -> None:
        self.param_stats: Dict[str, PolicyStats] = {}

    def record_outcome(self, param_key: str, reward: float, success: bool) -> None:
        """
        param_key: Policy 中某个可调字段的路径，如 "curiosity.min_concept_count"
        reward: 当前场景/实验的分数或归一化 reward
        """

        stats = self.param_stats.setdefault(param_key, PolicyStats())
        stats.attempts += 1
        stats.total_reward += float(reward)
        if success:
            stats.successes += 1

    def _get_by_path(self, policy: Any, path: str) -> Any:
        target = policy
        for name in path.split("."):
            if not hasattr(target, name):
                return None
            target = getattr(target, name)
        return target

    def _set_by_path(self, policy: Any, path: str, value: Any) -> None:
        parts = path.split(".")
        target = policy
        for name in parts[:-1]:
            if not hasattr(target, name):
                return
            target = getattr(target, name)
        if hasattr(target, parts[-1]):
            setattr(target, parts[-1], value)

    def propose_updates(self, policy: "AgentPolicy") -> Dict[str, Any]:
        """
        根据统计提出下一轮策略参数建议：
        - 对表现差的 param_key、小范围调节参数（如 ±10%）
        - 返回 param_key -> new_value 的 dict，供 PolicyPatch 或直接应用
        R15 可先实现简单规则（如 reward 越高，对应参数做小步上调/下调）。
        """

        updates: Dict[str, Any] = {}
        for path, stats in self.param_stats.items():
            if stats.attempts <= 0:
                continue
            current = self._get_by_path(policy, path)
            if current is None:
                continue

            avg_reward = stats.total_reward / max(1, stats.attempts)
            success_rate = stats.successes / max(1, stats.attempts)

            if isinstance(current, (int, float)):
                step = max(abs(float(current)) * 0.1, 0.1)
                direction = 0.0
                if avg_reward > 0.05:
                    direction = 1.0
                elif avg_reward < -0.05:
                    direction = -1.0
                elif success_rate > 0.7:
                    direction = 0.5
                elif success_rate < 0.3:
                    direction = -0.5
                if direction == 0.0:
                    continue
                new_value = float(current) + step * direction
                if "min_concept_count" in path:
                    new_value = max(1.0, new_value)
                updates[path] = new_value
            elif isinstance(current, bool):
                if success_rate < 0.4:
                    updates[path] = not current
            else:
                # 对字符串等类型仅在 reward 特别高时维持当前值（占位逻辑）
                if avg_reward > 0.8:
                    updates[path] = current
        return updates

    def apply_updates(self, policy: Any, updates: Dict[str, Any]) -> None:
        """将 propose_updates 生成的建议直接写入 policy。"""

        for path, value in updates.items():
            self._set_by_path(policy, path, value)
