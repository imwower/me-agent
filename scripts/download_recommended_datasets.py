#!/usr/bin/env python
from __future__ import annotations

"""从 ModelScope 下载推荐的中文多任务/多模态数据集（可选子集）。

说明：
    - 依赖 modelscope：未安装时提示 `pip install modelscope`；
    - 默认下载位置：data/modelscope/<dataset_id>；
    - 默认下载列表可用 --datasets 覆盖（逗号分隔）。

示例：
    python scripts/download_recommended_datasets.py
    python scripts/download_recommended_datasets.py --datasets iic/cmrc2018,damo/ChnSentiCorp
"""

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


DEFAULT_DATASETS: List[str] = [
    # 指令/QA/摘要/对话摘要
    "AI-ModelScope/belle_open_source_1M",  # BELLE 指令
    "iic/cmrc2018",  # 阅读理解
    "iic/drcd_zh",  # DRCD 中文 QA
    "AI-ModelScope/LCSTS",  # 新闻摘要
    "AI-ModelScope/CSDS",  # 对话摘要
    "damo/ChnSentiCorp",  # 情感分类
    "damo/tnews",  # 新闻分类
    # 多模态（图文/视觉问答/OCR）
    "damo/muge_retrieval",  # MUGE 图文检索
    "AI-ModelScope/coco_cn",  # 中文标注 COCO
    "damo/MTWI",  # 场景文字 OCR
    "AI-ModelScope/textcaps_cn",  # 中文图文描述
    "damo/DocVQA_cn",  # 文档 VQA（中文）
]


def _parse_dataset_list(raw: str | None) -> List[str]:
    if not raw:
        return DEFAULT_DATASETS
    return [item.strip() for item in raw.split(",") if item.strip()]


def _download(api, dataset_id: str, target_root: Path, revision: str = "master") -> None:
    """使用 ModelScope HubApi 下载数据集。"""

    logger.info("开始下载: %s", dataset_id)
    try:
        # HubApi.download_dataset 会将数据放到 local_dir/dataset_id 目录下
        api.download_dataset(dataset_id, revision=revision, local_dir=str(target_root))
    except Exception as exc:  # noqa: BLE001
        logger.error("下载失败: %s (%s)", dataset_id, exc)
    else:
        logger.info("完成: %s -> %s", dataset_id, target_root / dataset_id.replace("/", "_"))


def main() -> None:
    parser = argparse.ArgumentParser(description="下载推荐的中文多任务/多模态数据集（ModelScope）")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="自定义数据集列表，逗号分隔；默认使用内置推荐列表。",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/modelscope"),
        help="下载根目录，默认 data/modelscope",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="master",
        help="数据集版本，默认 master",
    )
    args = parser.parse_args()

    try:
        from modelscope.hub.api import HubApi  # type: ignore
    except Exception:
        logger.error("未找到 modelscope 库，请先运行: pip install modelscope")
        return

    dataset_list = _parse_dataset_list(args.datasets)
    target_root = args.target.expanduser()
    target_root.mkdir(parents=True, exist_ok=True)

    api = HubApi()
    for ds in dataset_list:
        _download(api, ds, target_root, revision=args.revision)

    logger.info("全部尝试完成。已下载的数据位于: %s", target_root)


if __name__ == "__main__":
    main()
