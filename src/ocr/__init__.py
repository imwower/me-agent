"""OCR 引擎封装模块。

在运行单元测试或尚未安装完整依赖时，可能无法正常导入
具体 OCR 实现（例如 cv2/pytesseract）。为了保证项目的基础
测试可以通过，这里对导入失败做了降级处理：
    - 若 PyTesseractOCREngine 导入失败，则将其设置为 None，
      调用方应在运行时检查并自行降级为“空 OCR”。
"""

from .base import BaseOCREngine  # noqa: F401

try:  # pragma: no cover - 降级路径主要用于缺失依赖时
    from .pytesseract_ocr import PyTesseractOCREngine  # type: ignore  # noqa: F401
except Exception:  # noqa: BLE001
    PyTesseractOCREngine = None  # type: ignore

__all__ = [
    "BaseOCREngine",
    "PyTesseractOCREngine",
]
