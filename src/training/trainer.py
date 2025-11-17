from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.dataloader import build_vqa_cn_stream
from src.training.losses import build_batch_texts_from_stream, compute_lm_step_loss
from src.training.optimizer import clip_gradients, create_optimizer, create_scheduler

logger = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    """根据配置与当前硬件自动选择 device。"""

    if device_str == "auto":
        if torch.backends.mps.is_available():
            logger.info("检测到 MPS，可使用 Apple Silicon GPU。")
            return torch.device("mps")
        if torch.cuda.is_available():
            logger.info("检测到 CUDA GPU。")
            return torch.device("cuda")
        return torch.device("cpu")

    if device_str == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cpu")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """通用训练入口（精简版 Demo）。

    当前实现针对中文 VQA 任务给出一个最小可运行训练循环：
        - 从配置中解析模型与数据参数；
        - 使用 Chinese-SimpleVQA 子集构造流式样本；
        - 仅使用文本解码器进行自回归训练（忽略视觉特征与桥接层），
          目的是验证数据与训练管线在 macOS + MPS 上可跑通；
        - 日志记录每若干步的平均损失。

    说明：
        - 这是一个“轻量自检”版本，不代表最终多任务/多头训练策略；
        - 后续可以在此基础上替换为完整的多模态模型与多任务损失。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="训练配置 YAML 路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = load_yaml(args.config)

    # 解析 defaults 列表，分别找到 model 与 dataset 配置
    defaults = cfg.get("defaults", [])
    model_cfg: Dict[str, Any] = {}
    dataset_cfg: Dict[str, Any] = {}

    config_dir = Path(args.config).resolve().parent
    for item in defaults:
        if "model" in item:
            model_cfg_path = config_dir / item["model"]
            model_cfg = load_yaml(str(model_cfg_path))
        if "dataset" in item:
            dataset_cfg_path = config_dir / item["dataset"]
            dataset_cfg = load_yaml(str(dataset_cfg_path))

    if not model_cfg:
        raise RuntimeError("未在配置中找到 model 配置，请检查 defaults。")

    train_cfg = model_cfg.get("training", {})

    device = _resolve_device(train_cfg.get("device", "auto"))
    logger.info("使用设备: %s", device)

    # 这里仅打印配置信息，作为 demo 占位。
    logger.info("模型配置: %s", model_cfg.get("model", {}))
    logger.info("数据配置: %s", dataset_cfg.get("dataset", {}))
    logger.info(
        "训练配置: lr=%.1e, max_steps=%d, batch_size=%d",
        train_cfg.get("lr", 1e-4),
        train_cfg.get("max_steps", 100),
        train_cfg.get("batch_size", 4),
    )

    # --------- 构造最小 VQA 训练数据流 ----------
    ds_cfg = dataset_cfg
    ds_name = ds_cfg.get("dataset", {}).get("name", "vqa_cn")
    if ds_name != "vqa_cn":
        logger.warning("当前 Demo 仅实现 vqa_cn 训练，将忽略 dataset.name=%s。", ds_name)

    def sample_stream() -> Iterable[Dict[str, Any]]:
        """对外暴露的样本流包装，内部使用 dataloader.build_vqa_cn_stream。"""

        return build_vqa_cn_stream(ds_cfg)

    # --------- 初始化文本解码器（作为最小模型） ----------
    text_cfg = model_cfg.get("model", {}).get("text_decoder", {})
    model_name = text_cfg.get("model_name", "gpt2")
    max_length = int(text_cfg.get("max_length", 64))

    logger.info("加载文本模型: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.train()

    optimizer = create_optimizer(model, train_cfg)

    max_steps = int(train_cfg.get("max_steps", 100))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    batch_size = int(train_cfg.get("batch_size", 4))

    scheduler = create_scheduler(
        optimizer,
        train_cfg,
        num_training_steps=max_steps,
    )

    global_step = 0
    running_loss = 0.0

    logger.info("开始最小 VQA 训练循环: max_steps=%d, batch_size=%d", max_steps, batch_size)

    # 简单批处理：每次从样本流中取 batch_size 条样本文本
    stream_iter = iter(sample_stream())
    while global_step < max_steps:
        batch_texts = build_batch_texts_from_stream(stream_iter, batch_size)

        if not batch_texts:
            logger.warning("本批次未采样到有效样本，结束训练循环。")
            break

        # 使用语言模型自回归损失（输入=标签）
        loss, scalar_loss = compute_lm_step_loss(
            model,
            tokenizer,
            batch_texts,
            device=device,
            max_length=max_length,
            grad_accum_steps=grad_accum,
        )
        loss.backward()
        running_loss += scalar_loss

        if (global_step + 1) % grad_accum == 0:
            clip_gradients(model, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % train_cfg.get("eval_steps", 50) == 0 or global_step == max_steps:
            avg_loss = running_loss / max(1, train_cfg.get("eval_steps", 50))
            logger.info("step=%d, 平均训练损失=%.4f", global_step, avg_loss)
            running_loss = 0.0

    logger.info("训练结束，总步数=%d", global_step)


if __name__ == "__main__":
    main()
