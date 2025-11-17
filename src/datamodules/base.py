from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OcrToken:
    """统一 OCR token 结构。"""

    id: str
    text: str
    bbox: List[int]  # [x1,y1,x2,y2]


@dataclass
class Region:
    """统一图像区域结构。"""

    id: str
    bbox: List[int]


@dataclass
class ChartElement:
    """统一图表元素结构。"""

    id: str
    type: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedSample:
    """统一后的多模态样本结构，对应题目中的 JSON schema。

    字段含义参考题目说明。
    """

    image: Any
    question: str
    answers: List[str]
    answerable: Optional[bool]
    evidence: Dict[str, Any]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为 JSON 友好的字典。"""

        return {
            "image": self.image,
            "question": self.question,
            "answers": list(self.answers),
            "answerable": self.answerable,
            "evidence": self.evidence,
            "meta": self.meta,
        }

