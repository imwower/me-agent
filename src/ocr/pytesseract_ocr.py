from __future__ import annotations

import logging
from typing import Any, Dict, List

import cv2
import numpy as np
import pytesseract

from .base import BaseOCREngine

logger = logging.getLogger(__name__)


class PyTesseractOCREngine(BaseOCREngine):
    """基于 pytesseract 的 OCR 引擎实现。

    说明：
        - 依赖本地安装 Tesseract；
        - 若环境中未安装 Tesseract 或不支持中文，可退回英文/数字识别；
        - 输出 token 列表，每个 token 附带简单矩形 bbox。
    """

    def __init__(self, lang: str = "chi_sim+eng") -> None:
        self.lang = lang

    def recognize(self, image: Any) -> List[Dict[str, Any]]:
        """对输入图像执行 OCR 并返回统一 token 格式。"""

        # 将输入转换为 OpenCV 图像
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            # 假设是 PIL.Image 或 numpy 数组
            if hasattr(image, "convert"):
                img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            else:
                img = np.array(image)

        if img is None:
            logger.warning("PyTesseractOCREngine: 无法读取图像，返回空结果。")
            return []

        data = pytesseract.image_to_data(
            img,
            lang=self.lang,
            output_type=pytesseract.Output.DICT,
        )

        tokens: List[Dict[str, Any]] = []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            if not text:
                continue
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            token = {
                "id": f"t{i}",
                "text": text,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
            }
            tokens.append(token)

        logger.info("PyTesseractOCREngine: 识别到 %d 个 token", len(tokens))
        return tokens

