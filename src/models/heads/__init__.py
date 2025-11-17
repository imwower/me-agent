"""多任务头部模块。"""

from .vqa_head import VQAHead  # noqa: F401
from .answerability_head import AnswerabilityHead  # noqa: F401
from .ocr_pointer_head import OCRPointerHead  # noqa: F401
from .chart_head import ChartHead  # noqa: F401

__all__ = [
  "VQAHead",
  "AnswerabilityHead",
  "OCRPointerHead",
  "ChartHead",
]

