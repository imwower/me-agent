from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _clean_line(text: str) -> str:
    """简单清洗文本行：去掉首尾空白。"""

    return text.strip()


def prepare_wiki_zh_2019(
    zip_path: Path,
    out_dir: Path,
    max_files: int | None = None,
) -> None:
    """将 wiki_zh_2019 语料转换为“每句一行、文档间空行”的预训练格式。

    输入：
        zip_path: data/wiki_zh_2019.zip（来自 brightmart/nlp_chinese_corpus）；
        out_dir: 输出目录，例如 data/wiki_zh_2019；
        max_files: 可选，只处理前 N 个小文件，便于快速调试。

    输出：
        out_dir/wiki_zh_sentences.txt
    """

    if not zip_path.exists():
        logger.warning("未找到 wiki_zh_2019.zip: %s，跳过该语料处理。", zip_path)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wiki_zh_sentences.txt"

    logger.info("开始处理 Wiki 语料: %s -> %s", zip_path, out_path)

    file_count = 0
    line_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf, out_path.open("w", encoding="utf-8") as out_f:
        # 过滤出实际包含内容的文件（排除目录名）
        member_names: Iterable[str] = [
            n for n in zf.namelist() if n.startswith("wiki_zh/") and not n.endswith("/")
        ]

        for name in member_names:
            file_count += 1
            if max_files is not None and file_count > max_files:
                break

            with zf.open(name) as f:
                for raw in f:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    text = obj.get("text") or ""
                    if not text:
                        continue

                    title = str(obj.get("title") or "").strip()
                    # 将正文按换行切分，过滤掉空行与与标题重复的行
                    for seg in text.split("\n"):
                        seg = _clean_line(seg)
                        if not seg:
                            continue
                        if seg == title:
                            continue
                        out_f.write(seg + "\n")
                        line_count += 1

                    # 文档之间增加一个空行
                    out_f.write("\n")

    logger.info(
        "Wiki 语料处理完成：共处理文件数=%d，写入行数=%d，输出=%s",
        file_count,
        line_count,
        out_path,
    )


def prepare_translation2019zh(
    zip_path: Path,
    out_dir: Path,
) -> None:
    """处理 translation2019zh 语料，将中英文句子对拆分为单独文件。

    输入：
        zip_path: data/translation2019zh.zip；
        out_dir: 输出目录，例如 data/translation2019zh。

    输出：
        out_dir/translation2019zh_zh.txt   # 每行一个中文句子
        out_dir/translation2019zh_en.txt   # 每行一个英文句子
    """

    if not zip_path.exists():
        logger.warning("未找到 translation2019zh.zip: %s，跳过该语料处理。", zip_path)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_zh = out_dir / "translation2019zh_zh.txt"
    out_en = out_dir / "translation2019zh_en.txt"

    logger.info("开始处理翻译语料: %s -> %s, %s", zip_path, out_zh, out_en)

    line_count = 0

    with zipfile.ZipFile(zip_path, "r") as zf, out_zh.open("w", encoding="utf-8") as zh_f, out_en.open(
        "w", encoding="utf-8"
    ) as en_f:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue

            logger.info("  处理文件: %s", name)
            with zf.open(name) as f:
                for raw in f:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    zh = _clean_line(str(obj.get("chinese") or ""))
                    en = _clean_line(str(obj.get("english") or ""))

                    if not zh and not en:
                        continue

                    if zh:
                        zh_f.write(zh + "\n")
                    if en:
                        en_f.write(en + "\n")

                    line_count += 1

    logger.info(
        "翻译语料处理完成：写入句对数=%d，输出中文文件=%s，英文文件=%s",
        line_count,
        out_zh,
        out_en,
    )


def main() -> None:
    """统一入口：处理 data/wiki_zh_2019.zip 与 data/translation2019zh.zip。"""

    base = Path(__file__).resolve().parent

    wiki_zip = base / "wiki_zh_2019.zip"
    wiki_out = base / "wiki_zh_2019"

    trans_zip = base / "translation2019zh.zip"
    trans_out = base / "translation2019zh"

    prepare_wiki_zh_2019(wiki_zip, wiki_out)
    prepare_translation2019zh(trans_zip, trans_out)


if __name__ == "__main__":
    main()

