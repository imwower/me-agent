"""受约束解码与指针复制相关模块。

当前实现提供：
- pointer_copy: 在证据池上进行简单的指针选择与置信度计算；
- trie: 前缀树结构，用于构建受限词表；
- constrained_decode: 基于证据与前缀树的简化受约束解码逻辑。
"""

from .pointer_copy import select_best_token  # noqa: F401
from .trie import Trie  # noqa: F401
from .constrained_decode import constrained_generate_from_evidence  # noqa: F401

__all__ = [
    "select_best_token",
    "Trie",
    "constrained_generate_from_evidence",
]

