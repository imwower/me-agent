#!/usr/bin/env python
from __future__ import annotations

"""在 CIFAR-100 上训练一个简单的 CNN 分类模型（实验性脚本）。

说明：
    - 完全基于本地 PyTorch / torchvision，不依赖 Hugging Face；
    - 主要用于为后续多模态对齐提供一个“图像编码器”示例；
    - 默认使用 ResNet18，并将权重保存到 outputs/cifar100_cnn 目录。

用法示例（在仓库根目录）：

    python scripts/train_cifar100_cnn.py \
      --data-root data/cifar100 \
      --epochs 10 \
      --batch-size 128 \
      --device auto
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def build_dataloaders(
    data_root: Path,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """构造 CIFAR-100 的训练与验证 DataLoader。"""

    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761],
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR100(
        root=str(data_root),
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = datasets.CIFAR100(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """在给定数据集上计算平均损失与准确率。"""

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += float(loss.item()) * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/cifar100",
        help="CIFAR-100 数据下载 / 存放路径。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="训练轮数。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="训练批大小。",
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

    device = resolve_device(args.device)
    logger.info("使用设备: %s", device)

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_dataloaders(data_root, args.batch_size)

    logger.info("构建 ResNet18 模型（num_classes=100）")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path("outputs/cifar100_cnn")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * images.size(0)
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_loss, val_acc = evaluate(model, test_loader, device)
        logger.info(
            "Epoch %d/%d: train_loss=%.4f, train_acc=%.4f, val_loss=%.4f, val_acc=%.4f",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        # 保存最佳模型 checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = output_dir / "best_resnet18_cifar100.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )
            logger.info("保存新的最佳模型 checkpoint: %s (val_acc=%.4f)", ckpt_path, val_acc)

    logger.info("训练结束，最佳验证准确率=%.4f", best_acc)


if __name__ == "__main__":
    main()

