from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseOCREngine(ABC):
    """OCR 引擎抽象基类。

    设计意图：
        - 对具体 OCR 实现（pytesseract / PaddleOCR 等）进行统一封装；
        - 输出格式统一为 token 列表，以便直接适配统一 schema 中的 ocr_tokens。
    """

    @abstractmethod
    def recognize(self, image: Any) -> List[Dict[str, Any]]:
        """对输入图像执行 OCR，返回 token 列表。

        每个 token 字典至少应包含：
            - id: str
            - text: str
            - bbox: [x1, y1, x2, y2]
        """

