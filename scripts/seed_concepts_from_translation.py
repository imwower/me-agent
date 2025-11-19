#!/usr/bin/env python
from __future__ import annotations

"""从中英文翻译语料中构建概念空间种子。

输入：
    - data/translation2019zh/translation2019zh_zh.txt
    - data/translation2019zh/translation2019zh_en.txt

行为：
    - 将每对 (中文句子, 英文句子) 视为同一概念的两个别名；
    - 使用 DummyEmbeddingBackend 将中文句子编码为概念向量；
    - 将概念与别名存入 ConceptSpace，并导出为 JSON 文件，供后续加载。

输出：
    - data/translation2019zh/concept_space_seed.json

注意：
    - 为避免一次性加载过多数据，脚本默认只处理前若干条（可通过参数调整）；
    - 生成的概念空间目前主要用于演示和后续扩展，不会自动在 Agent 中加载。
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# 确保可以从仓库根目录直接运行本脚本
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from me_core.alignment.concepts import ConceptSpace
from me_core.alignment.embeddings import DummyEmbeddingBackend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zh-path",
        type=str,
        default="data/translation2019zh/translation2019zh_zh.txt",
        help="中文句子文件路径。",
    )
    parser.add_argument(
        "--en-path",
        type=str,
        default="data/translation2019zh/translation2019zh_en.txt",
        help="英文句子文件路径。",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=50000,
        help="最多处理多少条中英句对，0 或负数表示处理全部。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/translation2019zh/concept_space_seed.json",
        help="概念空间种子输出路径（JSON）。",
    )
    args = parser.parse_args()

    zh_path = Path(args.zh_path)
    en_path = Path(args.en_path)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not zh_path.exists() or not en_path.exists():
        raise FileNotFoundError(
            f"找不到翻译语料文件：\n  zh: {zh_path}\n  en: {en_path}\n"
            "请先确保已运行 data/prepare_nlp_chinese_corpus.py。"
        )

    backend = DummyEmbeddingBackend(dim=32)
    space = ConceptSpace(similarity_threshold=0.9)

    logger.info(
        "开始从翻译语料构建概念空间种子：zh=%s, en=%s, max_pairs=%d",
        zh_path,
        en_path,
        args.max_pairs,
    )

    count = 0
    with zh_path.open("r", encoding="utf-8") as f_zh, en_path.open(
        "r", encoding="utf-8"
    ) as f_en:
        for zh_line, en_line in zip(f_zh, f_en):
            zh = zh_line.strip()
            en = en_line.strip()
            if not zh or not en:
                continue

            # 使用中文句子构建概念向量
            emb = backend.embed_text([zh])[0]
            node = space.get_or_create(emb, name_hint=zh[:32])
            # 为该概念注册中英文别名
            space.register_alias(node, zh)
            space.register_alias(node, en)

            count += 1
            if args.max_pairs > 0 and count >= args.max_pairs:
                break

            if count % 10000 == 0:
                logger.info("已处理句对数: %d", count)

    logger.info("共构建概念数: %d（处理句对数=%d）", len(space.all_concepts()), count)

    data = space.to_dict()
    with out_path.open("w", encoding="utf-8") as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=2)

    logger.info("概念空间种子已写入: %s", out_path)


if __name__ == "__main__":
    main()
