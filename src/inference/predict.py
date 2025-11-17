from __future__ import annotations

import json
from typing import Any, Dict, List


def predict_single(
    image: Any,
    question: str,
) -> Dict[str, Any]:
    """最小可运行推理入口（占位实现）。

    目标接口：
        输入：图像 + 中文问题；
        输出：统一 JSON：
            {
              "answer": "...",
              "answerable": true/false,
              "evidence": [{"type":"ocr_token|region|chart_element","id":"...","confidence":0.xx}],
              "confidence": 0.xx
            }

    当前实现：
        - 不加载真实模型，仅返回一个固定模板答案；
        - 主要用于打通 CLI / Notebook / API 的调用链路。
    """

    _ = image  # 当前 demo 未实际使用

    answer = f"这是一个示例回答：问题是「{question}」。"
    result = {
        "answer": answer,
        "answerable": True,
        "evidence": [],
        "confidence": 0.5,
    }
    # 确保可以被 JSON 序列化
    json.dumps(result, ensure_ascii=False)
    return result

