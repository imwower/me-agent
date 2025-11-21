from __future__ import annotations

import random
from typing import List

from me_core.config import AgentConfig
from me_core.policy import AgentPolicy
from me_core.policy.agents import AgentSpec
from me_core.teachers.types import PolicyPatch


def select_top(specs: List[AgentSpec], fitness_scores: List[float], k: int) -> List[AgentSpec]:
    paired = list(zip(specs, fitness_scores))
    paired.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in paired[:k]]


def mutate_policy(policy: AgentPolicy) -> AgentPolicy:
    new_policy = AgentPolicy(
        curiosity=policy.curiosity,
        dialogue=policy.dialogue,
        tools=policy.tools,
    )
    # 简单随机扰动：翻转启用开关或调整阈值
    if random.random() < 0.5:
        new_policy.curiosity.enabled = not new_policy.curiosity.enabled
    else:
        new_policy.curiosity.min_concept_count = max(1, new_policy.curiosity.min_concept_count + random.choice([-1, 1]))
    return new_policy


def mutate_config(config: AgentConfig) -> AgentConfig:
    new_cfg = AgentConfig(**config.__dict__)
    if random.random() < 0.5:
        new_cfg.enable_curiosity = not new_cfg.enable_curiosity
    if random.random() < 0.5:
        new_cfg.enable_introspection = not new_cfg.enable_introspection
    return new_cfg


def crossover_policy(p1: AgentPolicy, p2: AgentPolicy) -> AgentPolicy:
    child = AgentPolicy()
    child.curiosity = random.choice([p1.curiosity, p2.curiosity])
    child.dialogue = random.choice([p1.dialogue, p2.dialogue])
    child.tools = random.choice([p1.tools, p2.tools])
    return child


def apply_patches_to_spec(spec: AgentSpec, patches: List[PolicyPatch]) -> AgentSpec:
    from me_core.policy.applier import apply_policy_patches

    new_policy = apply_policy_patches(spec.policy, patches)
    return AgentSpec(id=spec.id, config=spec.config, policy=new_policy, backend_info=spec.backend_info)
