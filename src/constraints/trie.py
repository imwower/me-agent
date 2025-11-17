from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable


@dataclass
class TrieNode:
    """前缀树节点结构。"""

    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_end: bool = False


class Trie:
    """简单的前缀树实现，用于受限解码时约束可选前缀。

    使用场景：
        - 构建一个合法词典（单位/颜色/方向/数值格式等）；
        - 在 beam search 时，根据当前前缀筛选出下一个合法 token 列表。
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    @classmethod
    def from_words(cls, words: Iterable[str]) -> "Trie":
        trie = cls()
        for w in words:
            trie.insert(w)
        return trie

    def match_prefix(self, prefix: str) -> bool:
        """判断前缀是否仍存在于某条词路径上。"""

        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def is_word(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

