from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

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
    """通用训练入口（demo 版）。

    当前实现为占位训练循环，用于在小样本上验证数据管道与模型构造是否正确。
    实际多任务损失与完整模型拼装可在后续迭代中补全。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="训练配置 YAML 路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = load_yaml(args.config)

    # 解析嵌套 defaults（简化版：手动加载 model/dataset）
    model_cfg_path = cfg.get("defaults", [])[0].get("model", "configs/model_base.yaml")
    dataset_cfg_path = cfg.get("defaults", [])[0].get("dataset", None)

    base_dir = Path(args.config).resolve().parents[1]
    model_cfg = load_yaml(str(base_dir / model_cfg_path))
    if dataset_cfg_path:
        dataset_cfg = load_yaml(str(base_dir / dataset_cfg_path))
    else:
        dataset_cfg = {}

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

    # TODO: 在后续迭代中实现真正的多任务训练循环：
    #   - 构造 dataloader（根据 dataset.name 选择对应 datamodule）；
    #   - 初始化 vision encoder + text decoder + bridge + heads；
    #   - 仅对桥接层与头部启用梯度；
    #   - 使用混合精度与梯度累积；
    #   - 定期在验证集上评测并保存 best.ckpt。

    logger.info("当前 trainer.main 仅完成配置解析与设备选择，训练循环将在后续版本中补全。")


if __name__ == "__main__":
    main()

