from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from .schema import AgentPolicy, CuriosityPolicy, DialoguePolicyConfig, ToolSelectionPolicy


def load_policy_from_dict(d: Dict[str, Any]) -> AgentPolicy:
    def build(cls, data: Dict[str, Any]) -> Any:
        obj = cls()
        for k, v in data.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    curiosity = build(CuriosityPolicy, d.get("curiosity", {}))
    dialogue = build(DialoguePolicyConfig, d.get("dialogue", {}))
    tools = build(ToolSelectionPolicy, d.get("tools", {}))
    return AgentPolicy(curiosity=curiosity, dialogue=dialogue, tools=tools)


def policy_to_dict(policy: AgentPolicy) -> Dict[str, Any]:
    return asdict(policy)


def load_policy_from_file(path: str | None) -> AgentPolicy:
    if path is None:
        return AgentPolicy()
    data: Dict[str, Any] = {}
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    return load_policy_from_dict(data)


__all__ = ["AgentPolicy", "load_policy_from_dict", "policy_to_dict", "load_policy_from_file"]
