from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset

from .base import UnifiedSample

logger = logging.getLogger(__name__)


def _encode_image_field(example: Dict[str, Any]) -> Any:
    """将 Chinese-SimpleVQA 的图像字段转换为统一 image 字段。

    该数据集在 HuggingFace 上的字段通常包含 base64 / URL / image 对象等，
    为了兼容 streaming 模式，这里倾向于直接保留原始字段（例如 URL 或 PIL Image）。
    具体落地策略可根据需要在后续版本中扩展。
    """

    # 若存在 "image" 字段且为 image 对象，直接返回
    if "image" in example:
        return example["image"]

    # 若存在 "image_path" 或 "image_url" 等字段，则直接返回该路径/URL
    for key in ("image_path", "image_url", "url"):
        if key in example:
            return example[key]

    return None


def convert_example(example: Dict[str, Any]) -> UnifiedSample:
    """将 Chinese-SimpleVQA 的一条样本转换为统一 schema。"""

    image = _encode_image_field(example)
    question = str(example.get("final_question") or example.get("question") or "")
    answer = str(example.get("final_answer") or example.get("answer") or "")

    answers: List[str] = [answer] if answer else []

    unified = UnifiedSample(
        image=image,
        question=question,
        answers=answers,
        answerable=None,  # 该数据集未显式提供可答性标签
        evidence={
            "ocr_tokens": [],
            "regions": [],
            "chart_elements": [],
        },
        meta={
            "dataset": "csVQA",
            "split": "unknown",
        },
    )
    return unified


def load_chinese_simplevqa(
    split: str = "train",
    sample_ratio: float = 1.0,
    cache_dir: Optional[str] = None,
) -> Iterable[UnifiedSample]:
    """加载 Chinese-SimpleVQA 并转换为 UnifiedSample 序列。

    优先使用本地已下载文件（parquet/jsonl），若未找到则回退到
    HuggingFace streaming 加载。

    参数：
        split: 使用的数据集切分（train/validation/test 等），当前在本地文件模式下仅作日志用途；
        sample_ratio: 抽样比例（0,1]，用于在本地快速跑通实验；
        cache_dir: 本地缓存目录（同时用于推断 data/vqa_cn 根目录）。
    """

    logger.info(
        "加载 Chinese-SimpleVQA 数据集: split=%s, sample_ratio=%.3f",
        split,
        sample_ratio,
    )

    # 优先尝试从本地 data/vqa_cn 目录加载，避免每次都访问网络。
    base_dir = Path(cache_dir).resolve().parent if cache_dir else Path("data").resolve()
    local_root = base_dir / "vqa_cn"
    jsonl_path = local_root / "chinese_simplevqa.jsonl"
    parquet_path = local_root / "chinese_simplevqa.parquet"

    # 优先使用 jsonl，其次尝试 parquet
    if jsonl_path.exists():
        logger.info("从本地 jsonl 文件加载 Chinese-SimpleVQA: %s", jsonl_path)
        with jsonl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if sample_ratio < 1.0 and (i % int(1.0 / sample_ratio)) != 0:
                    continue
                try:
                    example = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                yield convert_example(example)
        return

    if parquet_path.exists():
        logger.info("从本地 parquet 文件加载 Chinese-SimpleVQA: %s", parquet_path)
        try:
            ds = load_dataset(
                "parquet",
                data_files={"train": str(parquet_path)},
                split="train",
                streaming=True,
            )
            for i, example in enumerate(ds):
                if sample_ratio < 1.0 and (i % int(1.0 / sample_ratio)) != 0:
                    continue
                yield convert_example(example)
            return
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "读取本地 parquet 失败，将回退到远程加载: %s",
                e,
            )
        return

    # 若本地文件不存在，则回退到 HuggingFace streaming 加载。
    logger.info(
        "未找到本地 Chinese-SimpleVQA 文件，将回退到 HuggingFace streaming 加载。"
    )

    ds = load_dataset(
        "OpenStellarTeam/Chinese-SimpleVQA",
        split=split,
        streaming=True,
        cache_dir=cache_dir,
    )

    for i, example in enumerate(ds):
        if sample_ratio < 1.0 and (i % int(1.0 / sample_ratio)) != 0:
            continue
        yield convert_example(example)
