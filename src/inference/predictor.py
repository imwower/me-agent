from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from PIL import Image

from src.models.multitask_model import MultiTaskModel
from .evidence import build_evidence_for_image_only
from .generate import evidence_first_generate

logger = logging.getLogger(__name__)

_MODEL_CACHE: Dict[Tuple[str, str], MultiTaskModel] = {}


def _resolve_device(device_str: str) -> torch.device:
    """根据字符串与当前硬件情况选择合适的 device。"""

    if device_str == "auto":
        if torch.backends.mps.is_available():
            logger.info("推理检测到 MPS，将使用 Apple Silicon GPU。")
            return torch.device("mps")
        if torch.cuda.is_available():
            logger.info("推理检测到 CUDA，将使用 GPU。")
            return torch.device("cuda")
        return torch.device("cpu")

    if device_str == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cpu")


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_multitask_model(
    model_config_path: str = "configs/model_base.yaml",
    device_str: str = "auto",
) -> Tuple[MultiTaskModel, torch.device]:
    """懒加载并缓存 MultiTaskModel，用于推理阶段。

    为避免每次调用都重新加载权重，这里在进程内做简单缓存：
        key = (model_config_path, device_str, checkpoint_path)
    """

    device = _resolve_device(device_str)
    # 尝试从 common.yaml 中解析 checkpoint 路径，若失败则回退到默认值
    common_cfg_path = Path("configs/common.yaml")
    checkpoint_path = None
    if common_cfg_path.exists():
        try:
            common_cfg = _load_yaml(str(common_cfg_path))
            train_cfg = common_cfg.get("train", {}) or {}
            ckpt_dir = Path(train_cfg.get("checkpoint_dir", "outputs/checkpoints"))
            ckpt_name = train_cfg.get("checkpoint_name", "multitask_best.pt")
            candidate = ckpt_dir / ckpt_name
            if candidate.exists():
                checkpoint_path = candidate
        except Exception:
            checkpoint_path = None

    cache_key = (
        str(Path(model_config_path).resolve()),
        str(device),
        str(checkpoint_path) if checkpoint_path is not None else "",
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], device

    cfg = _load_yaml(model_config_path)
    model = MultiTaskModel(cfg, device=device)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        logger.info("推理从 checkpoint 加载权重: %s", checkpoint_path)
    model.to(device)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    logger.info("推理加载 MultiTaskModel 完成: config=%s, device=%s", model_config_path, device)
    return model, device


def predict_single(
    image: Any,
    question: str,
    *,
    abstain_threshold: float = 0.4,
    pointer_min_conf: float = 0.35,
    use_model: bool = True,
    model_config: str = "configs/model_base.yaml",
    device: str = "auto",
) -> Dict[str, Any]:
    """面向外部调用的推理入口。

    输入：
        image: PIL.Image 或图像路径；
        question: 中文问题字符串。

    输出：
        {
          "answer": "...",
          "abstain": true/false,
          "confidence": 0.xx,
          "evidence": [
            {"type":"ocr|chart_elem|cell","id":"...","text":"...","confidence":0.xx}
          ]
        }
    """

    if isinstance(image, str):
        img_obj = Image.open(image).convert("RGB")
    else:
        img_obj = image

    evidence = build_evidence_for_image_only(img_obj)

    model: Optional[MultiTaskModel]
    if use_model:
        model, dev = load_multitask_model(model_config, device)
        # 目前 MultiTaskModel 主要提供可答性打分，
        # 证据选择仍由 constrained_decode 驱动。
    else:
        model = None

    result = evidence_first_generate(
        image=img_obj,
        question=question,
        evidence=evidence,
        abstain_threshold=abstain_threshold,
        pointer_min_conf=pointer_min_conf,
        model=model,
    )
    return result


def main() -> None:
    """命令行入口：单图单问题 Demo。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="图像路径")
    parser.add_argument("question", type=str, help="中文问题")
    parser.add_argument(
        "--abstain_threshold",
        type=float,
        default=0.4,
        help="拒答置信度阈值",
    )
    parser.add_argument(
        "--pointer_min_conf",
        type=float,
        default=0.35,
        help="指针最小置信度阈值",
    )
    parser.add_argument(
        "--no_model",
        action="store_true",
        help="仅使用基于证据的指针启发式，不加载多任务模型。",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_base.yaml",
        help="多任务模型配置文件路径（YAML）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推理设备: auto/mps/cpu/cuda。",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"找不到图像文件: {img_path}")

    result = predict_single(
        str(img_path),
        args.question,
        abstain_threshold=args.abstain_threshold,
        pointer_min_conf=args.pointer_min_conf,
        use_model=not args.no_model,
        model_config=args.model_config,
        device=args.device,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
