from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AgentConfig:
    use_dummy_embedding: bool = True
    embedding_backend_module: Optional[str] = None
    enable_curiosity: bool = True
    enable_introspection: bool = True
    episode_window: int = 1
    timeline_path: Optional[str] = None
    episodes_path: Optional[str] = None
    concepts_path: Optional[str] = None


def load_agent_config(path: Optional[str]) -> AgentConfig:
    if path is None:
        return AgentConfig()
    data: Dict[str, Any] = {}
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    cfg = AgentConfig()
    for field_name in cfg.__dataclass_fields__:  # type: ignore[attr-defined]
        if field_name in data:
            setattr(cfg, field_name, data[field_name])
    return cfg


__all__ = ["AgentConfig", "load_agent_config"]
