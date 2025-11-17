from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from PIL import Image

from .evidence import build_evidence_for_image_only
from .generate import evidence_first_generate

logger = logging.getLogger(__name__)


def predict_single(
    image: Any,
    question: str,
    *,
    abstain_threshold: float = 0.4,
    pointer_min_conf: float = 0.35,
) -> Dict[str, Any]:
    """面向外部调用的推理入口。

    输入：
        image: PIL.Image 或图像路径；
        question: 中文问题字符串。

    输出：
        {
          "answer": "...",
          "abstain": true/false,
          "confidence": 0.xx,
          "evidence": [
            {"type":"ocr|chart_elem|cell","id":"...","text":"...","confidence":0.xx}
          ]
        }
    """

    if isinstance(image, str):
        img_obj = Image.open(image).convert("RGB")
    else:
        img_obj = image

    evidence = build_evidence_for_image_only(img_obj)
    result = evidence_first_generate(
        image=img_obj,
        question=question,
        evidence=evidence,
        abstain_threshold=abstain_threshold,
        pointer_min_conf=pointer_min_conf,
    )
    return result


def main() -> None:
    """命令行入口：单图单问题 Demo。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="图像路径")
    parser.add_argument("question", type=str, help="中文问题")
    parser.add_argument(
        "--abstain_threshold",
        type=float,
        default=0.4,
        help="拒答置信度阈值",
    )
    parser.add_argument(
        "--pointer_min_conf",
        type=float,
        default=0.35,
        help="指针最小置信度阈值",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到图像文件: {img_path}")

    result = predict_single(
        str(img_path),
        args.question,
        abstain_threshold=args.abstain_threshold,
        pointer_min_conf=args.pointer_min_conf,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()

