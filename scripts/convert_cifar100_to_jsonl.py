"""将 CIFAR-100 小样本转换为本项目多模态 jsonl 格式（仅类别名作为文本）。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

try:
    from torchvision.datasets import CIFAR100  # type: ignore
    from torchvision import transforms  # type: ignore
except Exception as exc:  # pragma: no cover - 依赖外部库
    raise SystemExit(f"缺少 torchvision，请先安装：{exc}")


CIFAR100_ZH = {
    "apple": "苹果",
    "aquarium_fish": "热带鱼",
    "baby": "婴儿",
    "bear": "熊",
    "beaver": "河狸",
    "bed": "床",
    "bee": "蜜蜂",
    "beetle": "甲虫",
    "bicycle": "自行车",
    "bottle": "瓶子",
    "bowl": "碗",
    "boy": "男孩",
    "bridge": "桥",
    "bus": "公交车",
    "butterfly": "蝴蝶",
    "camel": "骆驼",
    "can": "罐头",
    "castle": "城堡",
    "caterpillar": "毛毛虫",
    "cattle": "牛",
    "chair": "椅子",
    "chimpanzee": "黑猩猩",
    "clock": "时钟",
    "cloud": "云",
    "cockroach": "蟑螂",
    "couch": "沙发",
    "crab": "螃蟹",
    "crocodile": "鳄鱼",
    "cup": "杯子",
    "dinosaur": "恐龙",
    "dolphin": "海豚",
    "elephant": "大象",
    "flatfish": "比目鱼",
    "forest": "森林",
    "fox": "狐狸",
    "girl": "女孩",
    "hamster": "仓鼠",
    "house": "房子",
    "kangaroo": "袋鼠",
    "keyboard": "键盘",
    "lamp": "台灯",
    "lawn_mower": "割草机",
    "leopard": "豹",
    "lion": "狮子",
    "lizard": "蜥蜴",
    "lobster": "龙虾",
    "man": "男人",
    "maple_tree": "枫树",
    "motorcycle": "摩托车",
    "mountain": "山",
    "mouse": "老鼠",
    "mushroom": "蘑菇",
    "oak_tree": "橡树",
    "orange": "橙子",
    "orchid": "兰花",
    "otter": "水獭",
    "palm_tree": "棕榈树",
    "pear": "梨",
    "pickup_truck": "皮卡",
    "pine_tree": "松树",
    "plain": "平原",
    "plate": "盘子",
    "poppy": "罂粟",
    "porcupine": "豪猪",
    "possum": "负鼠",
    "rabbit": "兔子",
    "raccoon": "浣熊",
    "ray": "鳐鱼",
    "road": "道路",
    "rocket": "火箭",
    "rose": "玫瑰",
    "sea": "海",
    "seal": "海豹",
    "shark": "鲨鱼",
    "shrew": "鼩鼱",
    "skunk": "臭鼬",
    "skyscraper": "摩天楼",
    "snail": "蜗牛",
    "snake": "蛇",
    "spider": "蜘蛛",
    "squirrel": "松鼠",
    "streetcar": "有轨电车",
    "sunflower": "向日葵",
    "sweet_pepper": "甜椒",
    "table": "桌子",
    "tank": "坦克",
    "telephone": "电话",
    "television": "电视",
    "tiger": "老虎",
    "tractor": "拖拉机",
    "train": "火车",
    "trout": "鳟鱼",
    "tulip": "郁金香",
    "turtle": "乌龟",
    "wardrobe": "衣柜",
    "whale": "鲸鱼",
    "willow_tree": "柳树",
    "wolf": "狼",
    "woman": "女人",
    "worm": "蠕虫",
}


def translate_label(name: str) -> str:
    return CIFAR100_ZH.get(name, name)


def load_cifar100(root: str, train: bool = True, limit_per_class: int = 10) -> List[Tuple[str, int]]:
    transform = transforms.Compose([])
    dataset = CIFAR100(root=root, train=train, download=False, transform=transform)
    indices_by_class: dict[int, List[int]] = {}
    for idx, (_, label) in enumerate(dataset):
        indices_by_class.setdefault(label, []).append(idx)
    selected: List[Tuple[str, int]] = []
    for label, idxs in indices_by_class.items():
        random.shuffle(idxs)
        for i in idxs[:limit_per_class]:
            subdir = Path(root) / ("train" if train else "test")
            subdir.mkdir(parents=True, exist_ok=True)
            img_path = subdir / f"{label}_{i}.png"
            img, _ = dataset[i]
            img.save(img_path)
            selected.append((str(img_path), label))
    return selected


def convert(root: str, output: str, limit_per_class: int = 10, sample: int = 10) -> None:
    samples = load_cifar100(root, train=True, limit_per_class=limit_per_class)
    random.shuffle(samples)
    samples = samples[:sample]
    class_names = CIFAR100(root=root, train=True, download=False).classes
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, (img_path, label_idx) in enumerate(samples):
            name = class_names[label_idx]
            zh = translate_label(name)
            filename = Path(img_path).name.replace(".png", f"_{zh}.png")
            # 重命名文件，添加中文方便查看
            target_path = Path(img_path).with_name(filename)
            Path(img_path).rename(target_path)
            rec = {
                "id": f"cifar-{idx}",
                "image_path": str(target_path),
                "text": zh,
                "labels": [zh],
                "task": "classification",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"写出 {len(samples)} 条到 {out_path}")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(description="将 CIFAR-100 转换为 jsonl（类别名作为文本）")
    parser.add_argument("--root", type=str, default="data/cifar100_raw", help="CIFAR100 原始存储目录")
    parser.add_argument("--output", type=str, default="data/benchmarks/cifar100_sample.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--limit-per-class", type=int, default=10, help="每类最多抽样多少张")
    parser.add_argument("--sample", type=int, default=10, help="最终输出样本总数")
    args = parser.parse_args()
    convert(args.root, args.output, limit_per_class=args.limit_per_class, sample=args.sample)


if __name__ == "__main__":  # pragma: no cover
    main()
