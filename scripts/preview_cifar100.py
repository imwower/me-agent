#!/usr/bin/env python
from __future__ import annotations

"""预览 CIFAR-100：保存前 8 张训练集图片及标签到 outputs/cifar100_preview.png。"""

from pathlib import Path

from torchvision import datasets, transforms, utils


def main() -> None:
    root = Path("data/cifar100")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = datasets.CIFAR100(root=str(root), train=True, download=False, transform=transforms.ToTensor())

    images = []
    captions = []
    for i in range(8):
        img, label = ds[i]
        images.append(img)
        captions.append(ds.classes[label])

    grid = utils.make_grid(images, nrow=4)
    out_path = out_dir / "cifar100_preview.png"
    utils.save_image(grid, out_path)

    labels_txt = out_dir / "cifar100_preview_labels.txt"
    labels_txt.write_text("\n".join(f"{i}: {name}" for i, name in enumerate(captions)), encoding="utf-8")

    print(f"已保存图片网格: {out_path}")
    print(f"对应标签: {labels_txt}")


if __name__ == "__main__":
    main()
