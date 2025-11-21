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

    # 加载策略：
    #   - 模型权重优先用本地 checkpoint，若不存在则用 base
    #   - tokenizer 优先用 checkpoint 中的文件，缺失时回退到 base
    model_path = ckpt if ckpt.exists() else base_model
    tokenizer_path = ckpt if (ckpt / "tokenizer.json").exists() else base_model

    if model_path is None:
        raise FileNotFoundError(f"找不到检查点 {ckpt}，也未提供 --base 基础模型。")
    if tokenizer_path is None:
        raise FileNotFoundError("找不到 tokenizer 文件，请通过 --base 指定基础模型。")

    print(f"[加载模型] {model_path}")
    print(f"[加载分词器] {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path)
    except Exception:
        # 若加载失败，再次强制回退基础模型
        if base_model is None:
            raise
        print("[警告] 无法从本地检查点加载 tokenizer/模型，回退到基础模型。")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(base_model)

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
