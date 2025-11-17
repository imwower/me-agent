"""推理与统一 JSON 输出模块。

当前提供：
- predictor.predict_single: 面向外部调用的主推理接口；
- evidence / generate: 证据检索与两段式生成的内部组件。
"""

from .predictor import predict_single  # noqa: F401
from .evidence import build_evidence_for_image_only  # noqa: F401
from .generate import evidence_first_generate  # noqa: F401

__all__ = [
    "predict_single",
    "build_evidence_for_image_only",
    "evidence_first_generate",
]
