from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from . import (
    chart_scores,
    compute_answerability_ap,
    ocr_vqa_scores,
    vqa_accuracy,
)

logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        logger.error("找不到文件: %s", path)
        return []
    items: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def main() -> None:
    """汇总评测入口。

    约定：
        - pred/refs 均为 JSONL，字段符合统一 schema；
        - 对 VQA / OCR-VQA / Chart-QA 分别计算核心指标；
        - 生成 coverage-risk 曲线示例图与简单 CSV 摘要。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="预测结果 JSONL 路径")
    parser.add_argument("--ref", type=str, required=True, help="参考标注 JSONL 路径")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/eval",
        help="评测结果输出目录",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    preds = _load_jsonl(args.pred)
    refs = _load_jsonl(args.ref)
    if not preds or not refs:
        logger.error("预测或参考文件为空，无法评测。")
        return

    # 将参考标注按 id 索引
    ref_map: Dict[str, Dict[str, Any]] = {ex["id"]: ex for ex in refs if "id" in ex}

    # 按 task 汇总
    vqa_pred: List[str] = []
    vqa_ref: List[List[str]] = []
    ocr_pred: List[str] = []
    ocr_ref: List[str] = []
    chart_pred_vals: List[float] = []
    chart_ref_vals: List[float] = []
    answerability_scores: List[float] = []
    answerability_labels: List[int] = []

    for pred in preds:
        ex_id = pred.get("id")
        if ex_id is None or ex_id not in ref_map:
            continue
        ref = ref_map[ex_id]
        task = ref.get("task")
        answer = str(pred.get("answer") or "")

        # 可答性
        abstain = bool(pred.get("abstain"))
        # 参考 answerable 若为 True 则 label=1，否则为0（含不可答/缺失）
        ref_answerable = ref.get("answerable")
        if isinstance(ref_answerable, bool):
            label = 1 if ref_answerable else 0
            answerability_labels.append(label)
            # 使用 (1 - abstain) 作为简单可答性得分
            answerability_scores.append(0.0 if abstain else 1.0)

        if task == "vqa_cn":
            vqa_pred.append(answer)
            vqa_ref.append(list(ref.get("answers") or []))
        elif task in ("ocr_vqa", "docvqa_cn"):
            ocr_pred.append(answer)
            # 简化：仅取第一个参考答案
            ans_list = list(ref.get("answers") or [])
            ocr_ref.append(ans_list[0] if ans_list else "")
        elif task == "chart_qa":
            try:
                pred_val = float(answer)
                true_list = list(ref.get("answers") or [])
                true_val = float(true_list[0]) if true_list else 0.0
                chart_pred_vals.append(pred_val)
                chart_ref_vals.append(true_val)
            except ValueError:
                continue

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # VQA
    vqa_acc = vqa_accuracy(vqa_pred, vqa_ref) if vqa_pred else 0.0
    # OCR
    ocr_hit, ocr_ed = ocr_vqa_scores(ocr_pred, ocr_ref) if ocr_pred else (0.0, 0.0)
    # Chart
    chart_em, chart_mape = chart_scores(chart_pred_vals, chart_ref_vals) if chart_pred_vals else (0.0, 0.0)
    # Answerability
    ans_ap = (
        compute_answerability_ap(answerability_scores, answerability_labels)
        if answerability_labels
        else 0.0
    )

    # 简单 coverage-risk 曲线：以不同阈值过滤 abstain 比例与错误率（占位实现）
    coverages = [1.0, 0.8, 0.6, 0.4, 0.2]
    risks = [1.0 - vqa_acc for _ in coverages]
    plt.figure()
    plt.plot(coverages, risks, marker="o")
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - VQA Acc)")
    plt.title("Coverage-Risk Curve (示意)")
    plt.grid(True)
    png_path = out_dir / "coverage_risk.png"
    plt.savefig(png_path, dpi=150)

    # 写 CSV 摘要
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("task,metric,value\n")
        f.write(f"vqa_cn,acc,{vqa_acc:.4f}\n")
        f.write(f"ocr_vqa,hit_rate,{ocr_hit:.4f}\n")
        f.write(f"ocr_vqa,avg_edit_distance,{ocr_ed:.4f}\n")
        f.write(f"chart_qa,em,{chart_em:.4f}\n")
        f.write(f"chart_qa,mape,{chart_mape:.4f}\n")
        f.write(f"answerability,ap,{ans_ap:.4f}\n")

    logger.info("评测完成，结果已写入: %s 与 %s", csv_path, png_path)


if __name__ == "__main__":
    main()

