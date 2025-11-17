from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)


def load_mmbench_cn(path: str) -> List[Dict[str, str]]:
    """加载 MMBench-CN 数据集占位实现。

    说明：
        - 当前不尝试自动下载 MMBench-CN；
        - 若指定路径不存在，则打印官方链接与放置说明，并返回空列表；
        - 若存在，则假设为 JSONL/CSV，并留给用户在后续迭代中扩展解析逻辑。
    """

    p = Path(path)
    if not p.exists():
        logger.warning(
            "未找到 MMBench-CN 数据文件: %s\n"
            "请访问官方仓库获取数据，并按照 README 中的说明将文件放置到指定路径。\n"
            "官方链接示例：https://github.com/open-compass/MMBench",
            path,
        )
        return []

    # 目前仅返回空列表，解析逻辑留待后续补充
    logger.info("检测到 MMBench-CN 数据文件: %s，但当前仅实现占位加载。", path)
    return []


def score_mmbench_cn(
    preds: Iterable[str],
    refs: Iterable[str],
) -> float:
    """MMBench-CN 评分占位实现。

    当前仅用于保持接口完整，直接返回 0.0。
    """

    _ = list(preds), list(refs)
    logger.info("当前版本暂未实现 MMBench-CN 真实评分逻辑。")
    return 0.0

