from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from src.datamodules.chinese_simplevqa import load_chinese_simplevqa
from src.datamodules.xfund_zh import load_xfund_zh
from src.datamodules.chart.chartqa import load_chartqa

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


def build_ocr_vqa_stream(dataset_cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """构造 OCR-VQA/文档 VQA 任务的样本流。

    当前实现基于 XFUND-zh 数据集（见 configs/dataset_ocr_vqa_cn.yaml），
    重点验证“读字/结构化文档”的抽取式问答能力。
    """

    ds_cfg = dataset_cfg.get("dataset", {})
    hf_cfgs: List[Dict[str, Any]] = ds_cfg.get("datasets", [])
    if not hf_cfgs:
        raise RuntimeError("dataset.datasets 为空，无法构造 ocr_vqa 样本流。")

    hf_cfg = hf_cfgs[0]
    split = hf_cfg.get("split", "train")
    sample_ratio = float(hf_cfg.get("sample_ratio", 1.0))
    cache_dir = ds_cfg.get("cache_dir", "data/cache")
    max_samples = int(ds_cfg.get("max_samples", 1000))

    logger.info(
        "构造 ocr_vqa 样本流: split=%s, sample_ratio=%.3f, max_samples=%d, cache_dir=%s",
        split,
        sample_ratio,
        max_samples,
        cache_dir,
    )

    def _stream() -> Iterator[Dict[str, Any]]:
        count = 0
        for uni in load_xfund_zh(
            split=split,
            sample_ratio=sample_ratio,
            cache_dir=cache_dir,
        ):
            yield uni.to_dict()
            count += 1
            if count >= max_samples:
                break

    return _stream()


def build_chart_qa_stream(dataset_cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """构造图表问答任务的样本流。

    当前实现基于 ChartQA 数据集，使用 HuggingFace streaming 接口加载。
    """

    ds_cfg = dataset_cfg.get("dataset", {})
    hf_cfgs: List[Dict[str, Any]] = ds_cfg.get("datasets", [])
    if not hf_cfgs:
        raise RuntimeError("dataset.datasets 为空，无法构造 chart_qa 样本流。")

    hf_cfg = hf_cfgs[0]
    split = hf_cfg.get("split", "train")
    sample_ratio = float(hf_cfg.get("sample_ratio", 1.0))
    cache_dir = ds_cfg.get("cache_dir", "data/cache")
    max_samples = int(ds_cfg.get("max_samples", 1000))

    logger.info(
        "构造 chart_qa 样本流: split=%s, sample_ratio=%.3f, max_samples=%d, cache_dir=%s",
        split,
        sample_ratio,
        max_samples,
        cache_dir,
    )

    def _stream() -> Iterator[Dict[str, Any]]:
        count = 0
        for uni in load_chartqa(
            split=split,
            sample_ratio=sample_ratio,
            cache_dir=cache_dir,
        ):
            yield uni.to_dict()
            count += 1
            if count >= max_samples:
                break

    return _stream()


class MultiTaskBatchIterator:
    """多任务批次迭代器。

    根据 multitask.mix_ratio 配置，在多个任务之间交替采样 batch。
    当前支持任务：
        - "vqa_cn"
        - "ocr_vqa"
        - "chart_qa"

    返回的 batch 结构：
        {
          "task": <任务名>,
          "samples": [UnifiedSample.to_dict(), ...]  # 长度约等于 batch_size
        }
    """

    def __init__(
        self,
        dataset_cfg_map: Dict[str, Dict[str, Any]],
        mix_ratio: Dict[str, int],
        batch_size: int,
    ) -> None:
        self.dataset_cfg_map = dataset_cfg_map
        self.mix_ratio = {k: int(v) for k, v in mix_ratio.items() if v > 0}
        self.batch_size = batch_size

        # 为每个任务绑定构造样本流的函数
        self.builders: Dict[str, Callable[[Dict[str, Any]], Iterable[Dict[str, Any]]]] = {}
        if "vqa_cn" in self.dataset_cfg_map:
            self.builders["vqa_cn"] = build_vqa_cn_stream
        if "ocr_vqa" in self.dataset_cfg_map:
            self.builders["ocr_vqa"] = build_ocr_vqa_stream
        if "chart_qa" in self.dataset_cfg_map:
            self.builders["chart_qa"] = build_chart_qa_stream

        # 构造任务轮询序列
        self.tasks: List[str] = []
        for task, r in self.mix_ratio.items():
            if task in self.builders:
                self.tasks.extend([task] * r)
            else:
                logger.warning("mix_ratio 中包含未注册任务 %s，将被忽略。", task)

        if not self.tasks:
            raise RuntimeError("MultiTaskBatchIterator: 没有可用任务，请检查 mix_ratio 与数据配置。")

        self._task_idx = 0

        # 统计每个任务被采样的次数，便于在训练日志中观察 mix_ratio 是否生效
        self._sample_counts: Dict[str, int] = {task: 0 for task in self.builders}

        # 为每个任务初始化样本流迭代器
        self._streams: Dict[str, Iterable[Dict[str, Any]]] = {}
        self._iters: Dict[str, Iterator[Dict[str, Any]]] = {}
        for task in self.builders:
            cfg = self.dataset_cfg_map.get(task, {})
            stream = self.builders[task](cfg)
            self._streams[task] = stream
            self._iters[task] = iter(stream)

    def __iter__(self) -> "MultiTaskBatchIterator":
        return self

    def _next_sample_for_task(self, task: str) -> Optional[Dict[str, Any]]:
        """从指定任务流中获取下一个样本，必要时自动重建流。

        若连续两次尝试都无法获取样本，则返回 None。
        """

        for _ in range(2):
            it = self._iters[task]
            try:
                return next(it)
            except StopIteration:
                # 重建流
                cfg = self.dataset_cfg_map.get(task, {})
                stream = self.builders[task](cfg)
                self._streams[task] = stream
                self._iters[task] = iter(stream)
        return None

    def __next__(self) -> Dict[str, Any]:
        if not self.tasks:
            raise StopIteration

        # 按 mix_ratio 轮询选择当前任务
        task = self.tasks[self._task_idx % len(self.tasks)]
        self._task_idx += 1

        samples: List[Dict[str, Any]] = []
        for _ in range(self.batch_size * 2):
            sample = self._next_sample_for_task(task)
            if sample is None:
                # 该任务流无法提供样本，将其从任务列表中移除
                logger.warning("任务 %s 无可用样本，将从轮询中移除。", task)
                self.tasks = [t for t in self.tasks if t != task]
                if not self.tasks:
                    raise StopIteration
                # 换下一个任务重试
                task = self.tasks[self._task_idx % len(self.tasks)]
                continue
            samples.append(sample)
            if len(samples) >= self.batch_size:
                break

        if not samples:
            raise StopIteration

        # 累积采样次数，用于训练时的统计与调试
        self._sample_counts[task] = self._sample_counts.get(task, 0) + 1

        return {
            "task": task,
            "samples": samples,
        }

    def get_sample_counts(self) -> Dict[str, int]:
        """返回当前各任务被采样的批次数。

        该信息可用于训练日志中观察多任务采样是否符合 mix_ratio 预期。
        """

        return dict(self._sample_counts)
