from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_charty_zh_stub(root_dir: str = "data/charty_zh") -> None:
    """ChartY-zh 准备脚本占位实现。

    说明：
        - 实际数据集可能需要从官方站点下载并解压；
        - 这里仅确保目录结构存在，方便后续用户放置数据。
    """

    root = Path(root_dir)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "val").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)

    logger.info("确保 ChartY-zh 本地目录结构存在: %s", root_dir)


if __name__ == "__main__":
    prepare_charty_zh_stub()

