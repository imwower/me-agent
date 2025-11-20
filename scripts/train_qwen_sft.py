#!/usr/bin/env python
from __future__ import annotations

"""基于 Chinese-Qwen3-235B-2507-Distill-data-110k-SFT 的最小 SFT 训练脚本。

特点：
    - 使用 Hugging Face transformers + Trainer；
    - 读取 ModelScope 下载到本地的 JSONL，多轮对话字段为 messages；
    - 默认模型：Qwen2-0.5B-Instruct（可通过 --model 覆盖）；
    - 仅做示范，未启用 LoRA/梯度累积优化，需自行根据显存调整超参。

依赖：
    pip install -r requirements.txt  # transformers / datasets / accelerate / modelscope

示例：
    python scripts/train_qwen_sft.py \\
      --data-file data/modelscope/swift___chinese-qwen3-235_b-2507-distill-data-110k-sft/qwen3_235b_2507_distill_110k.jsonl \\
      --model Qwen/Qwen2-0.5B-Instruct \\
      --output outputs/qwen3_sft_demo
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def format_messages(messages: List[Dict[str, str]], sep: str = "\n") -> str:
    """将多轮 messages 列表拼成模型可训练的纯文本。"""

    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    return sep.join(parts).strip()


def _auto_find_jsonl(root: Path) -> Optional[Path]:
    """在 root 下自动查找 jsonl 数据文件（含无扩展名的下载缓存）。"""

    candidates = list(root.rglob("*.jsonl"))
    # 兼容 ModelScope downloads 生成的无扩展名文件：存在同名 .json 元信息
    for p in root.rglob("*"):
        if p.is_file() and "." not in p.name:
            meta = p.with_name(p.name + ".json")
            if meta.exists():
                candidates.append(p)

    if not candidates:
        return None
    return sorted(candidates, key=lambda p: len(str(p)))[0]


def build_dataset(path: Path, max_samples: int | None = None):
    ds = load_dataset("json", data_files={"train": str(path)}, split="train")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    ds = ds.map(lambda x: {"text": format_messages(x["messages"])}, remove_columns=ds.column_names, num_proc=4)
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="JSONL 数据路径（包含 messages 字段）。默认自动在 data/modelscope 下搜索 *.jsonl。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="基础模型名称或本地路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qwen3_sft_demo"),
        help="模型输出目录。",
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数。")
    parser.add_argument("--batch-size", type=int, default=2, help="每卡批大小。")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率。")
    parser.add_argument("--max-samples", type=int, default=20000, help="可选：截取前 N 条样本，加速演示。")
    parser.add_argument("--fp16", action="store_true", help="启用 FP16 训练（需 GPU 支持）。")
    args = parser.parse_args()

    data_path: Path
    if args.data_file is None:
        auto_root = Path("data/modelscope").expanduser()
        data_path = _auto_find_jsonl(auto_root)  # type: ignore[assignment]
        if data_path is None:
            raise FileNotFoundError("未找到数据文件，请通过 --data-file 指定 jsonl 路径。")
    else:
        data_path = args.data_file.expanduser()
        if not data_path.exists():
            raise FileNotFoundError(f"找不到数据文件: {data_path}")

    print(f"[数据] 加载 {data_path}")
    train_ds = build_dataset(data_path, max_samples=args.max_samples)
    print(f"[数据] 样本数: {len(train_ds)}")

    print(f"[模型] 加载 {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        result = tokenizer(
            batch["text"],
            truncation=True,
            max_length=1024,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.model)

    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=args.fp16,
        gradient_accumulation_steps=1,
        remove_unused_columns=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    print("[训练] 开始")
    trainer.train()
    print("[训练] 完成，模型保存在:", args.output)


if __name__ == "__main__":
    main()
