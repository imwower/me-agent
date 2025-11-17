from __future__ import annotations

import logging
from typing import Any, Dict, List

from PIL import Image

from src.ocr import PyTesseractOCREngine

logger = logging.getLogger(__name__)


def _ensure_pil_image(image: Any) -> Image.Image:
    """将任意输入转换为 PIL.Image，用于 OCR 等后续处理。"""

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    # 尝试从 numpy array 或其他格式构造
    return Image.fromarray(image)


def collect_ocr_evidence(image: Any) -> List[Dict[str, Any]]:
    """对图像运行 OCR，生成 ocr_tokens 列表。

    若 OCR 引擎不可用，则返回空列表，并记录警告日志。
    """

    if PyTesseractOCREngine is None:
        logger.warning("当前环境未安装 pytesseract/cv2，将不返回 OCR 证据。")
        return []

    pil_img = _ensure_pil_image(image)
    engine = PyTesseractOCREngine()
    tokens = engine.recognize(pil_img)
    # 确保每个 token 都包含 conf 字段
    for t in tokens:
        t.setdefault("conf", 0.9)
    return tokens


def build_evidence_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """从统一 schema 的样本字典中抽取 evidence 字段。

    若字段不存在则返回空结构。
    """

    ev = sample.get("evidence") or {}
    return {
        "ocr_tokens": list(ev.get("ocr_tokens") or []),
        "regions": list(ev.get("regions") or []),
        "chart_elements": list(ev.get("chart_elements") or []),
        "table_cells": list(ev.get("table_cells") or []),
    }


def build_evidence_for_image_only(image: Any) -> Dict[str, Any]:
    """仅从原始图像构建证据池（主要用于 CLI demo）。"""

    ocr_tokens = collect_ocr_evidence(image)
    return {
        "ocr_tokens": ocr_tokens,
        "regions": [],
        "chart_elements": [],
        "table_cells": [],
    }

