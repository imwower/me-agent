from __future__ import annotations

from typing import Any, List

from .schema import AgentPolicy
from ..teachers.types import PolicyPatch


def _set_by_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    target = obj
    for name in parts[:-1]:
        if not hasattr(target, name):
            return
        target = getattr(target, name)
    if hasattr(target, parts[-1]):
        setattr(target, parts[-1], value)


def apply_policy_patches(policy: AgentPolicy, patches: List[PolicyPatch]) -> AgentPolicy:
    for patch in patches:
        _set_by_path(policy, patch.path, patch.value)
    return policy


__all__ = ["apply_policy_patches"]
