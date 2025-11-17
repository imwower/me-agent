"""OCR 引擎封装模块。"""

from .base import BaseOCREngine  # noqa: F401
from .pytesseract_ocr import PyTesseractOCREngine  # noqa: F401

__all__ = [
    "BaseOCREngine",
    "PyTesseractOCREngine",
]

