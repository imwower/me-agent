from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentRoleConfig:
    id: str
    description: str
    tools: List[str] = field(default_factory=list)
    initial_policy_overrides: Dict[str, Any] = field(default_factory=dict)


ROLES_DEFAULT: Dict[str, AgentRoleConfig] = {
    "planner": AgentRoleConfig(id="planner", description="规划步骤与任务拆解", tools=["echo", "time", "brain_infer"]),
    "coder": AgentRoleConfig(id="coder", description="负责代码改写与工具调用", tools=["read_file", "write_file", "apply_patch"]),
    "tester": AgentRoleConfig(id="tester", description="负责运行测试/实验", tools=["run_tests", "run_training"]),
    "brain": AgentRoleConfig(id="brain", description="负责调用 self-snn / BrainTools", tools=["dump_brain_graph", "eval_brain_energy", "eval_brain_memory", "brain_infer"]),
    "critic": AgentRoleConfig(id="critic", description="审阅结果与 Teacher 建议", tools=[]),
}


__all__ = ["AgentRoleConfig", "ROLES_DEFAULT"]
