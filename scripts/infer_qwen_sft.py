#!/usr/bin/env python
from __future__ import annotations

"""使用训练好的 Qwen SFT 检查点做简单对话生成（命令行）。"""

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(user_text: str) -> str:
    return f"user: {user_text}\nassistant: "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/qwen3_sft_m2"),
        help="训练好的本地检查点路径（需包含 tokenizer 模型文件或通过 --base 指定基础模型）。",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="基础模型名称或路径，用于缺失 tokenizer/权重时回退。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="生成最大新 token 数。",
    )
    args = parser.parse_args()

    ckpt = args.model
    base_model = args.base

    # 强制使用微调检查点；若缺失 tokenizer 文件则报错，避免悄然回退基础模型。
    model_path = ckpt
    tokenizer_path = ckpt

    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型检查点: {model_path}")
    if not (ckpt / "tokenizer.json").exists() and base_model is None:
        raise FileNotFoundError("检查点缺少 tokenizer 文件，请确保保存完整或通过 --base 指定 tokenizer 来源。")

    # 若需要，可单独指定基础 tokenizer，但模型权重必须来自微调检查点
    tokenizer_load = tokenizer_path if (ckpt / "tokenizer.json").exists() else base_model
    if tokenizer_load is None:
        raise FileNotFoundError("未找到 tokenizer.json，且未提供 --base tokenizer。")

    print(f"[加载模型权重] {model_path}")
    print(f"[加载分词器] {tokenizer_load}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    while True:
        try:
            user_text = input("你: ").strip()
        except EOFError:
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            print("再见～")
            break

        prompt = build_prompt(user_text)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取 assistant 段落
        if "assistant:" in reply:
            reply = reply.split("assistant:", 1)[1].strip()
        print(f"Agent: {reply}")


if __name__ == "__main__":
    main()
