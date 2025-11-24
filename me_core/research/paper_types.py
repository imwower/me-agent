from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Section:
    title: str
    content: str
    subsections: List["Section"] = field(default_factory=list)


@dataclass
class PaperDraft:
    title: str
    abstract: str
    sections: List[Section]
    meta: dict[str, Any] = field(default_factory=dict)


__all__ = ["Section", "PaperDraft"]
