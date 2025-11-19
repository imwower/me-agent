#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CharLM(nn.Module):
    """与 train_char_lm_nlp_corpus.py 中相同结构的字符级 LSTM 语言模型。"""

    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512, num_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
) -> int:
    """根据最后一步 logits 采样下一个字符索引。"""

    if temperature <= 0:
        # 贪心
        return int(torch.argmax(logits).item())

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return int(idx.item())


def generate_text(
    model: CharLM,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    device: torch.device,
    prompt: str,
    max_new_chars: int,
    block_size: int,
    temperature: float,
) -> str:
    model.eval()

    # 将提示编码为索引序列，忽略不在词表中的字符
    seq = [stoi[ch] for ch in prompt if ch in stoi]
    if not seq:
        # 若提示中没有任何已知字符，则随机选一个起始字符
        import random

        seq = [random.choice(list(stoi.values()))]

    for _ in range(max_new_chars):
        # 取最后 block_size 个字符作为输入
        context = seq[-block_size:]
        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

        with torch.no_grad():
            logits, _ = model(x)
        last_logits = logits[0, -1, :]  # [vocab_size]
        next_idx = sample_next_token(last_logits, temperature=temperature)
        seq.append(next_idx)

    # 解码为文本
    text = "".join(itos[i] for i in seq)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/char_lm_nlp_corpus/char_lm_last.pt",
        help="字符级语言模型 checkpoint 路径。",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="outputs/char_lm_nlp_corpus/vocab.json",
        help="字符词表 JSON 路径（由训练脚本保存的 stoi 字典）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推理设备: auto/cpu/cuda/mps。",
    )
    parser.add_argument(
        "--max-new-chars",
        type=int,
        default=200,
        help="在提示之后生成的最大字符数。",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="上下文窗口大小（需与训练时 block_size 一致）。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="采样温度，越大越随机，0 或以下表示贪心解码。",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    logger.info("使用设备: %s", device)

    model_path = Path(args.model_path)
    vocab_path = Path(args.vocab_path)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型 checkpoint: {model_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"找不到字符词表: {vocab_path}")

    # 加载词表
    with vocab_path.open("r", encoding="utf-8") as f:
        stoi: Dict[str, int] = json.load(f)
    itos: Dict[int, str] = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)
    logger.info("加载字符词表成功，大小=%d", vocab_size)

    # 初始化模型并加载参数
    state = torch.load(model_path, map_location=device)
    model = CharLM(vocab_size=vocab_size)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device)

    print("【字符级语言模型交互】输入中文前缀，模型将续写；输入 exit/quit 退出。")

    try:
        while True:
            try:
                prompt = input("你说> ").strip()
            except EOFError:
                print()
                break

            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit", "q"}:
                break

            generated = generate_text(
                model=model,
                stoi=stoi,
                itos=itos,
                device=device,
                prompt=prompt,
                max_new_chars=args.max_new_chars,
                block_size=args.block_size,
                temperature=args.temperature,
            )
            print(f"模型> {generated}")
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()

