#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _prepare_hf_env(hf_endpoint: Optional[str], http_proxy: Optional[str], https_proxy: Optional[str]) -> None:
    """根据参数配置 HuggingFace 端点与代理，并启用 hf-xet 插件（如可用）。"""

    if hf_endpoint:
        # 同时设置 HF_ENDPOINT / HF_HUB_ENDPOINT，提升兼容性
        os.environ["HF_ENDPOINT"] = hf_endpoint
        os.environ["HF_HUB_ENDPOINT"] = hf_endpoint
        logger.info("已设置 HuggingFace 端点为: %s", hf_endpoint)

    # 默认开启 hf-xet 插件，让 huggingface_hub 在可用时自动使用 xet-core 加速下载。
    if not os.getenv("HF_HUB_ENABLE_HF_XET"):
        os.environ["HF_HUB_ENABLE_HF_XET"] = "1"
        logger.info("已启用 HF_HUB_ENABLE_HF_XET=1（如安装 hf-xet/xet-core，则会自动加速下载）。")

    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
        os.environ["http_proxy"] = http_proxy
        logger.info("已设置 HTTP 代理: %s", http_proxy)

    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy
        os.environ["https_proxy"] = https_proxy
        logger.info("已设置 HTTPS 代理: %s", https_proxy)


def _download_split(root: Path, split: str) -> None:
    """使用 hf CLI 下载指定 split 的 Chinese-SimpleVQA 数据集。

    说明：
        - 实际调用的是 `hf download`，带 `--repo-type dataset` 与 `--resume-download`；
        - CLI 会自动处理断点续传与进度显示；
        - Chinese-SimpleVQA 目前以单个 `chinese_simplevqa.parquet` 存放全部样本，
          因此对不同 split，这里实际上都是下载同一个底层文件。
    """

    target_dir = root
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "使用 hf CLI 下载 Chinese-SimpleVQA（逻辑 split=%s），目标目录=%s",
        split,
        target_dir,
    )

    cmd = [
        "hf",
        "download",
        "OpenStellarTeam/Chinese-SimpleVQA",
        "--repo-type",
        "dataset",
        "--local-dir",
        str(target_dir),
    ]

    logger.info("执行命令: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as e:  # noqa: BLE001
        logger.error(
            "未找到 `hf` 命令，请先安装 huggingface_hub CLI 并确保其在 PATH 中：pip install -U 'huggingface_hub[cli]'"
        )
        raise e
    except subprocess.CalledProcessError as e:  # noqa: BLE001
        logger.error("hf download 命令执行失败，返回码=%d", e.returncode)
        raise e

    logger.info("split=%s 下载完成，数据保存在目录: %s", split, target_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/vqa_cn",
        help="数据根目录，将在其中创建下载好的数据文件（例如 chinese_simplevqa.parquet）",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=["train"],
        help='需要下载的切分名称列表，例如: --splits train validation test，默认仅 "train"',
    )
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="可选，HuggingFace 国内镜像/私有端点，默认: https://hf-mirror.com",
    )
    parser.add_argument(
        "--http-proxy",
        type=str,
        default=None,
        help="可选，HTTP 代理地址，例如 http://127.0.0.1:7890",
    )
    parser.add_argument(
        "--https-proxy",
        type=str,
        default=None,
        help="可选，HTTPS 代理地址，例如 http://127.0.0.1:7890",
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # 先根据参数配置国内镜像 / 代理，再进行实际下载
    _prepare_hf_env(
        hf_endpoint=args.hf_endpoint,
        http_proxy=args.http_proxy,
        https_proxy=args.https_proxy,
    )

    split_names: List[str] = args.splits
    logger.info(
        "目标根目录: %s, 计划下载的 splits: %s（CLI 实际下载整个数据集仓库，与 splits 设置关系不大）",
        root,
        ", ".join(split_names),
    )

    for split in split_names:
        try:
            _download_split(root, split)
        except Exception as e:  # noqa: BLE001
            logger.warning("下载或写入 split=%s 失败，将跳过。错误: %s", split, e)


if __name__ == "__main__":
    main()
