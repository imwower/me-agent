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
        help="训练好的本地检查点路径。",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="可选：若本地检查点不存在，使用的基础模型（如 Qwen/Qwen2-0.5B-Instruct）。",
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
    base_model = args.base or ckpt

    if not ckpt.exists() and args.base is None:
        raise FileNotFoundError(f"找不到检查点 {ckpt}，请通过 --base 指定基础模型或先训练。")

    print(f"[加载模型] {base_model}")
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
