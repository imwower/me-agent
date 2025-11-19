#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CharDataset(Dataset):
    """简单字符级语言模型数据集。

    将多个文本文件拼接成一个长字符串，根据 block_size 构造 (input, target) 序列：
        input:  x[t : t+block]
        target: x[t+1 : t+block+1]
    """

    def __init__(
        self,
        texts: List[str],
        block_size: int,
        max_chars: int | None = None,
    ) -> None:
        super().__init__()
        full_text = "\n".join(texts)
        if max_chars is not None and max_chars > 0:
            full_text = full_text[:max_chars]

        # 构建字符级词表
        chars = sorted(set(full_text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        logger.info("字符表大小: %d", self.vocab_size)

        self.block_size = block_size

        # 将全文编码为索引序列
        self.data = torch.tensor([self.stoi[ch] for ch in full_text], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


class CharLM(nn.Module):
    """简单字符级 LSTM 语言模型。"""

    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512, num_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def load_text_files(paths: List[Path], max_chars_per_file: int | None = None) -> List[str]:
    texts: List[str] = []
    for p in paths:
        if not p.exists():
            logger.warning("跳过不存在的文本文件: %s", p)
            continue
        logger.info("读取文本文件: %s", p)
        with p.open("r", encoding="utf-8") as f:
            if max_chars_per_file is None or max_chars_per_file <= 0:
                texts.append(f.read())
            else:
                texts.append(f.read(max_chars_per_file))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki-path",
        type=str,
        default="data/wiki_zh_2019/wiki_zh_sentences.txt",
        help="Wiki 语料清洗后的文本路径。",
    )
    parser.add_argument(
        "--trans-path",
        type=str,
        default="data/translation2019zh/translation2019zh_zh.txt",
        help="translation2019zh 中文句子文本路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/char_lm_nlp_corpus",
        help="训练输出目录（保存 checkpoint 与词表）。",
    )
    parser.add_argument(
        "--max-chars-per-file",
        type=int,
        default=2_000_000,
        help="每个文件最多读取多少字符，0 或负数表示全部读取。",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="训练序列长度。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="训练 batch 大小。",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=2000,
        help="训练步数（梯度更新次数）。",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="训练设备: auto/cpu/cuda/mps。",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 设备选择
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info("使用设备: %s", device)

    # 加载文本
    texts = load_text_files(
        [Path(args.wiki_path), Path(args.trans_path)],
        max_chars_per_file=args.max_chars_per_file,
    )
    if not texts:
        raise RuntimeError("未成功加载任何文本，请检查路径与数据文件。")

    # 构建数据集与 DataLoader
    dataset = CharDataset(texts=texts, block_size=args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 保存字符表，便于后续推理使用
    vocab_path = out_dir / "vocab.json"
    import json

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(dataset.stoi, f, ensure_ascii=False, indent=2)
    logger.info("已保存字符表到: %s", vocab_path)

    # 初始化模型与优化器
    model = CharLM(vocab_size=dataset.vocab_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    log_interval = 100
    total_loss = 0.0

    logger.info(
        "开始字符级语言模型训练: steps=%d, batch_size=%d, block_size=%d, vocab_size=%d",
        args.num_steps,
        args.batch_size,
        args.block_size,
        dataset.vocab_size,
    )

    model.train()
    data_iter = iter(dataloader)
    while global_step < args.num_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, dataset.vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        global_step += 1
        total_loss += float(loss.item())

        if global_step % log_interval == 0:
            avg_loss = total_loss / log_interval
            ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
            logger.info("step=%d, loss=%.4f, ppl=%.2f", global_step, avg_loss, ppl)
            total_loss = 0.0

            # 保存 checkpoint
            ckpt_path = out_dir / "char_lm_last.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": global_step,
                    "config": {
                        "vocab_size": dataset.vocab_size,
                        "block_size": args.block_size,
                    },
                },
                ckpt_path,
            )
            logger.info("已保存 checkpoint 到: %s", ckpt_path)

    logger.info("训练完成，总步数=%d", global_step)


if __name__ == "__main__":
    main()

