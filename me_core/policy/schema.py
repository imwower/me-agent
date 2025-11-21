from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CuriosityPolicy:
    enabled: bool = True
    min_concept_count: int = 3
    text_only_penalty: float = 0.5


@dataclass
class DialoguePolicyConfig:
    use_self_description: bool = True
    max_recent_concepts: int = 3
    style: Literal["default", "curious", "formal"] = "default"


@dataclass
class ToolSelectionPolicy:
    prefer_high_success_rate: bool = True
    min_calls_for_stat: int = 3


@dataclass
class AgentPolicy:
    curiosity: CuriosityPolicy = field(default_factory=CuriosityPolicy)
    dialogue: DialoguePolicyConfig = field(default_factory=DialoguePolicyConfig)
    tools: ToolSelectionPolicy = field(default_factory=ToolSelectionPolicy)
