from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml
from PIL import Image
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


def build_texts_for_task(task: str, samples: List[Dict[str, Any]]) -> List[str]:
    """根据任务类型构造训练文本。

    当前阶段仍然只训练纯文本自回归损失，但不同任务可以使用稍有差异的提示模版，
    便于未来扩展到多头多损失。
    """

    texts: List[str] = []
    for ex in samples:
        q = ex.get("question") or ""
        ans_list = ex.get("answers") or []
        if not ans_list:
            # 没有参考答案时跳过该样本
            continue
        a = ans_list[0]

        if task == "vqa_cn":
            prefix = "问题："
            mid = "\n答案："
        elif task == "ocr_vqa":
            prefix = "图片文字相关问题："
            mid = "\n答案："
        elif task == "chart_qa":
            prefix = "基于图表回答问题："
            mid = "\n答案："
        else:
            prefix = "问题："
            mid = "\n答案："

        texts.append(f"{prefix}{q}{mid}{a}")

    return texts


def _maybe_collect_pil_images(samples: List[Dict[str, Any]]) -> List[Image.Image]:
    """从样本中尽量收集可用的 PIL.Image。

    说明：
        - 视觉前向目前仅用于调试与形状检查，不参与损失；
        - 若某个样本缺少 image 路径或无法打开，则忽略。
    """

    images: List[Image.Image] = []
    for ex in samples:
        img_path = ex.get("image")
        if not img_path:
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        images.append(img)
    return images


def main() -> None:
    """训练入口。

    支持两种模式：
        1) 单任务 VQA Demo（原有行为，config 为 train_vqa_cn.yaml 等）；
        2) 多任务文本 Demo（config 为 common.yaml，包含 multitask 配置），
           使用 MultiTaskModel 但当前仅训练文本侧自回归损失。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="训练配置 YAML 路径")
    parser.add_argument(
        "--multitask",
        action="store_true",
        help="启用多任务训练模式（使用 configs/common.yaml 样式配置）。",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_yaml(args.config)

    # 若配置中包含 multitask 字段，或显式指定 --multitask，则走多任务路径
    if args.multitask or "multitask" in cfg:
        run_multitask_training(cfg)
        return

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


def run_multitask_training(cfg: Dict[str, Any]) -> None:
    """多任务训练入口（简化文本版）。

    当前实现目标：
        - 读取 common.yaml 风格配置（含 device/optimizer/scheduler/train/multitask/datasets）；
        - 使用 MultiTaskModel 作为统一文本模型（视觉与桥接层暂不参与训练，只初始化）；
        - 根据 multitask.mix_ratio 在 VQA / OCR-VQA / Chart-QA 之间轮流采样 batch；
        - 对所有任务统一使用文本自回归损失（后续可扩展为多头多损失）。
    """

    logging.basicConfig(level=logging.INFO)

    # 解析模型配置与数据配置路径
    config_dir = Path(".")
    model_cfg_path = config_dir / cfg.get("model_config", "configs/model_base.yaml")
    model_cfg = load_yaml(str(model_cfg_path))

    datasets_cfg_paths: Dict[str, str] = cfg.get("datasets", {}) or {}
    dataset_cfg_map: Dict[str, Dict[str, Any]] = {}
    for task, path in datasets_cfg_paths.items():
        dataset_cfg_map[task] = load_yaml(str(Path(path)))

    device = _resolve_device(cfg.get("device", "auto"))
    logger.info("多任务训练使用设备: %s", device)

    # 初始化 MultiTaskModel（当前主要使用文本侧，视觉侧只做前向检查）
    from src.models.multitask_model import MultiTaskModel  # 延迟导入，避免在无依赖环境下失败

    model = MultiTaskModel(model_cfg, device=device)
    model.text_model.train()

    # 构造多任务样本流
    from src.training.dataloader import MultiTaskBatchIterator

    multitask_cfg = cfg.get("multitask", {}) or {}
    mix_ratio = multitask_cfg.get(
        "mix_ratio",
        {"vqa_cn": 4, "ocr_vqa": 3, "chart_qa": 3},
    )

    train_cfg_raw = cfg.get("train", {}) or {}
    opt_cfg = cfg.get("optimizer", {}) or {}
    sched_cfg = cfg.get("scheduler", {}) or {}

    # 将 common.yaml 中的配置转换为 optimizer/ scheduler 所需的键
    train_cfg: Dict[str, Any] = {
        "lr": opt_cfg.get("lr", 2.0e-4),
        "weight_decay": opt_cfg.get("weight_decay", 0.01),
        "scheduler": {"name": sched_cfg.get("name", "linear")},
        "max_steps": int(train_cfg_raw.get("max_steps", 400)),
        "gradient_accumulation_steps": int(train_cfg_raw.get("grad_accum_steps", 4)),
        "batch_size": int(train_cfg_raw.get("batch_size_per_device", 1)),
    }

    max_steps = int(train_cfg["max_steps"])
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    batch_size = int(train_cfg["batch_size"])

    # MultiTaskBatchIterator 期望任务名为 vqa_cn / ocr_vqa / chart_qa，
    # 因此需要从 dataset_cfg_map 中挑选对应配置。
    dataset_cfg_for_mt: Dict[str, Dict[str, Any]] = {}
    if "vqa_cn" in dataset_cfg_map:
        dataset_cfg_for_mt["vqa_cn"] = dataset_cfg_map["vqa_cn"]
    if "ocr_vqa" in dataset_cfg_map:
        dataset_cfg_for_mt["ocr_vqa"] = dataset_cfg_map["ocr_vqa"]
    if "chart_qa" in dataset_cfg_map:
        dataset_cfg_for_mt["chart_qa"] = dataset_cfg_map["chart_qa"]

    if not dataset_cfg_for_mt:
        raise RuntimeError("多任务训练未找到任何可用数据集配置，请检查 common.yaml 中的 datasets 字段。")

    batch_iterator = MultiTaskBatchIterator(
        dataset_cfg_map=dataset_cfg_for_mt,
        mix_ratio=mix_ratio,
        batch_size=batch_size,
    )

    # checkpoint 配置：保存最优模型，供推理与评测使用
    ckpt_dir = Path(train_cfg_raw.get("checkpoint_dir", "outputs/checkpoints"))
    ckpt_name = train_cfg_raw.get("checkpoint_name", "multitask_best.pt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / ckpt_name
    best_loss = float("inf")

    optimizer = create_optimizer(model, train_cfg)
    scheduler = create_scheduler(optimizer, train_cfg, num_training_steps=max_steps)

    global_step = 0
    running_loss = 0.0

    # 按任务维护一个滑动窗口内的 loss，用于更细粒度观察训练情况
    task_loss_window: Dict[str, List[float]] = defaultdict(list)
    log_interval = int(train_cfg_raw.get("log_interval", 50))

    logger.info(
        "开始多任务训练循环: max_steps=%d, batch_size=%d, mix_ratio=%s",
        max_steps,
        batch_size,
        mix_ratio,
    )

    while global_step < max_steps:
        try:
            batch = next(batch_iterator)
        except StopIteration:
            logger.info("多任务迭代器耗尽，提前结束训练。")
            break

        task = batch["task"]
        samples = batch["samples"]

        # 文本侧：根据任务构造提示模版
        texts = build_texts_for_task(task, samples)
        if not texts:
            logger.warning("任务 %s 本批次未采样到有效文本样本，将跳过。", task)
            continue

        # 视觉侧：前向形状检查，同时为可答性分支提供可选视觉摘要
        pil_images = _maybe_collect_pil_images(samples)
        if pil_images:
            with torch.no_grad():
                vision_feats = model.encode_images(pil_images)
                bridged = model.bridge_vision(vision_feats)
            if global_step % log_interval == 0:
                logger.info(
                    "[视觉] step=%d, task=%s, vision_feats=%s, bridged=%s",
                    global_step,
                    task,
                    tuple(vision_feats.shape),
                    tuple(bridged.shape),
                )

        # 从样本中提取可答性标签（若存在）
        answerable_labels: List[Optional[bool]] = []
        for ex in samples:
            label = ex.get("answerable", None)
            if isinstance(label, bool):
                answerable_labels.append(label)
            else:
                answerable_labels.append(None)

        # 当前阶段：文本 LM + 可答性二分类的联合损失，
        # 若提供 pil_images，则在可答性分支内部融合桥接后的视觉摘要，实现简单多模态训练。
        answer_weight = float(train_cfg_raw.get("answerability_weight", 1.0))
        joint_loss = model.forward_with_answerability(
            texts,
            answerable_labels=answerable_labels,
            images=pil_images,
            answerability_weight=answer_weight,
        )

        # 若当前任务为 OCR-VQA，则额外加入指针式 OCR 拷贝损失
        if task == "ocr_vqa":
            ocr_weight = float(train_cfg_raw.get("ocr_pointer_weight", 1.0))
            ocr_loss = model.compute_ocr_pointer_loss_from_samples(samples)
            if ocr_loss is not None:
                joint_loss = joint_loss + ocr_weight * ocr_loss

        # 若当前任务为 Chart-QA，则额外加入图表元素/数值抽取损失
        if task == "chart_qa":
            chart_weight = float(train_cfg_raw.get("chart_loss_weight", 1.0))
            chart_loss = model.compute_chart_loss_from_samples(samples)
            if chart_loss is not None:
                joint_loss = joint_loss + chart_weight * chart_loss

        loss = joint_loss / grad_accum
        loss.backward()
        running_loss += float(loss.item())
        task_loss_window[task].append(float(loss.item()))

        if (global_step + 1) % grad_accum == 0:
            clip_gradients(model.text_model, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % log_interval == 0 or global_step == max_steps:
            avg_loss = running_loss / max(1, min(global_step, log_interval))

            # 计算各任务最近窗口的平均损失
            task_loss_info = []
            for t, losses in task_loss_window.items():
                if losses:
                    task_loss_info.append(f"{t}_avg_loss={sum(losses) / len(losses):.4f}")

            # 多任务采样次数
            sample_counts = batch_iterator.get_sample_counts()

            logger.info(
                "step=%d, 最近平均损失=%.4f, 当前任务=%s, %s, sample_counts=%s",
                global_step,
                avg_loss,
                task,
                " ".join(task_loss_info),
                sample_counts,
            )

            running_loss = 0.0
            # 简单重置窗口，避免无限增长
            task_loss_window = defaultdict(list)

            # 以最近平均损失作为简单“验证指标”，保存最优 checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    model.state_dict(),
                    best_ckpt_path,
                )
                logger.info("保存新的最优多任务模型 checkpoint: %s (loss=%.4f)", best_ckpt_path, best_loss)

    logger.info("多任务训练结束，总步数=%d", global_step)


if __name__ == "__main__":
    main()
