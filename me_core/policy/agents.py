from __future__ import annotations

from dataclasses import dataclass

from me_core.config import AgentConfig, load_agent_config
from me_core.policy import AgentPolicy, load_policy_from_file


@dataclass
class AgentSpec:
    id: str
    config: AgentConfig
    policy: AgentPolicy
    backend_info: dict[str, object] | None = None


def load_agent_spec_from_files(
    agent_id: str,
    config_path: str | None,
    policy_path: str | None,
) -> AgentSpec:
    cfg = load_agent_config(config_path)
    policy = load_policy_from_file(policy_path)
    return AgentSpec(id=agent_id, config=cfg, policy=policy, backend_info=None)
