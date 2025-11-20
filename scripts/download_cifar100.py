#!/usr/bin/env python
from __future__ import annotations

"""下载并解压 CIFAR-100（Python 版）到本地 data 目录。

默认使用官方链接：https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
如需使用镜像，可通过命令行参数 --url 指定。

用法（仓库根目录执行）：

    python scripts/download_cifar100.py \
      --output data/cifar100 \
      --url https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
"""

import argparse
import logging
import ssl
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    """限制解压路径，避免 tar slip。"""

    def _members() -> Iterable[tarfile.TarInfo]:
        for member in tar.getmembers():
            member_path = (path / member.name).resolve()
            if not str(member_path).startswith(str(path.resolve())):
                raise RuntimeError(f"检测到潜在路径穿越: {member.name}")
            yield member

    tar.extractall(path, members=_members())


def download_file(url: str, dest: Path, verify: bool = True) -> None:
    """流式下载文件到 dest。"""

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("开始下载: %s", url)

    context = None if verify else ssl._create_unverified_context()
    if not verify:
        logger.warning("已关闭证书校验（仅在受信网络使用）。")

    with urlopen(url, context=context) as resp, dest.open("wb") as f:
        total_str = resp.getheader("Content-Length")
        total_size = int(total_str) if total_str else None
        downloaded = 0
        last_report = -5

        while True:
            chunk = resp.read(8 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            if total_size:
                percent = int(downloaded * 100 / total_size)
                if percent - last_report >= 5:
                    last_report = percent
                    logger.info("  已下载 %d%% (%0.1f MB/%0.1f MB)", percent, downloaded / 1e6, total_size / 1e6)

    logger.info("下载完成，保存到: %s (%.1f MB)", dest, downloaded / 1e6)


def extract_archive(archive_path: Path, out_dir: Path) -> None:
    """解压并移动到目标目录。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:
        tmp_path = Path(tmp_dir)
        logger.info("开始解压: %s", archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract(tar, tmp_path)

        extracted = tmp_path / "cifar-100-python"
        if not extracted.exists():
            raise RuntimeError("未在压缩包中找到 cifar-100-python 目录，解压失败。")

        target = out_dir / "cifar-100-python"
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), str(target))

    logger.info("解压完成，数据目录: %s", out_dir / "cifar-100-python")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并解压 CIFAR-100 数据集（Python 版）")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cifar100"),
        help="数据存放目录（默认 data/cifar100）",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        help="下载链接，可自定义镜像。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载并覆盖已有数据。",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="关闭 HTTPS 证书校验（不推荐，仅在受信网络调试）。",
    )
    args = parser.parse_args()

    data_root = args.output.expanduser()
    data_root.mkdir(parents=True, exist_ok=True)

    target_dir = data_root / "cifar-100-python"
    archive_path = data_root / "cifar-100-python.tar.gz"

    if target_dir.exists() and not args.force:
        logger.info("检测到已存在的数据目录，跳过下载: %s", target_dir)
        return

    if archive_path.exists() and args.force:
        archive_path.unlink()

    with tempfile.NamedTemporaryFile(dir=data_root, suffix=".tmp", delete=False) as tmp_f:
        tmp_path = Path(tmp_f.name)

    try:
        download_file(args.url, tmp_path, verify=not args.insecure)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_path), archive_path)
        extract_archive(archive_path, data_root)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    logger.info("全部完成。")


if __name__ == "__main__":
    main()
