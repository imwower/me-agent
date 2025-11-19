"""推理与统一 JSON 输出模块（当前仅保留 OCR 证据构建）。"""

from __future__ import annotations

from .evidence import build_evidence_for_image_only, collect_ocr_evidence, build_evidence_from_sample

__all__ = ["build_evidence_for_image_only", "collect_ocr_evidence", "build_evidence_from_sample"]
