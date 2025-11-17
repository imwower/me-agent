from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Iterator, List

from src.datamodules.chinese_simplevqa import load_chinese_simplevqa

logger = logging.getLogger(__name__)


def build_vqa_cn_stream(dataset_cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """构造中文 VQA（vqa_cn）任务的样本流。

    该函数读取 dataset 配置，并返回一个可以被多次迭代的生成器：
        - 内部使用 HuggingFace datasets 的 streaming 接口加载
          Chinese-SimpleVQA；
        - 根据 sample_ratio 进行下采样；
        - 最多返回 max_samples 条样本。

    返回的每个元素均为统一 schema 的字典形式
    （即 UnifiedSample.to_dict() 的结果）。
    """

    ds_cfg = dataset_cfg.get("dataset", {})
    hf_cfgs: List[Dict[str, Any]] = ds_cfg.get("datasets", [])
    if not hf_cfgs:
        raise RuntimeError("dataset.datasets 为空，无法构造 vqa_cn 样本流。")

    hf_cfg = hf_cfgs[0]
    split = hf_cfg.get("split", "train")
    sample_ratio = float(hf_cfg.get("sample_ratio", 1.0))
    cache_dir = ds_cfg.get("cache_dir", "data/cache")
    max_samples = int(ds_cfg.get("max_samples", 1000))

    logger.info(
        "构造 vqa_cn 样本流: split=%s, sample_ratio=%.3f, max_samples=%d, cache_dir=%s",
        split,
        sample_ratio,
        max_samples,
        cache_dir,
    )

    def _stream() -> Iterator[Dict[str, Any]]:
        count = 0
        for uni in load_chinese_simplevqa(
            split=split,
            sample_ratio=sample_ratio,
            cache_dir=cache_dir,
        ):
            yield uni.to_dict()
            count += 1
            if count >= max_samples:
                break

    return _stream()

