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

    # 按 task 汇总整体指标
    vqa_pred: List[str] = []
    vqa_ref: List[List[str]] = []
    ocr_pred: List[str] = []
    ocr_ref: List[str] = []
    chart_pred_vals: List[float] = []
    chart_ref_vals: List[float] = []
    answerability_scores: List[float] = []
    answerability_labels: List[int] = []

    # 按数据集维度再细分一层（task.dataset）
    vqa_pred_by_ds: Dict[str, List[str]] = defaultdict(list)
    vqa_ref_by_ds: Dict[str, List[List[str]]] = defaultdict(list)
    ocr_pred_by_ds: Dict[str, List[str]] = defaultdict(list)
    ocr_ref_by_ds: Dict[str, List[str]] = defaultdict(list)
    chart_pred_by_ds: Dict[str, List[float]] = defaultdict(list)
    chart_ref_by_ds: Dict[str, List[float]] = defaultdict(list)

    # 为 coverage-risk 曲线准备 per-example 级别信息
    coverage_items: List[Tuple[float, bool, bool]] = []

    # 证据质量相关统计（OCR 证据命中率、图表元素覆盖率）
    ocr_evi_hits = 0
    ocr_evi_total = 0
    chart_evi_hits = 0
    chart_evi_total = 0

    for pred in preds:
        ex_id = pred.get("id")
        if ex_id is None or ex_id not in ref_map:
            continue
        ref = ref_map[ex_id]
        task = ref.get("task")
        meta = ref.get("meta") or {}
        dataset = str(meta.get("dataset") or "unknown")
        answer = str(pred.get("answer") or "")
        conf = float(pred.get("confidence", 0.0))

        # 可答性
        abstain = bool(pred.get("abstain"))
        # 参考 answerable 若为 True 则 label=1，否则为0（含不可答/缺失）
        ref_answerable = ref.get("answerable")
        if isinstance(ref_answerable, bool):
            label = 1 if ref_answerable else 0
            answerability_labels.append(label)
            # 使用推理阶段给出的 confidence 作为可答性置信度
            answerability_scores.append(conf)

        correct = False

        # 预测给出的证据列表（用于 OCR/Chart 证据质量评估）
        pred_evidence = pred.get("evidence") or []

        if task == "vqa_cn":
            vqa_pred.append(answer)
            vqa_ref.append(list(ref.get("answers") or []))
            vqa_pred_by_ds[dataset].append(answer)
            vqa_ref_by_ds[dataset].append(list(ref.get("answers") or []))

            # 逐样本正确性（简化版 VQA-Acc）
            norm_pred = answer.strip().lower()
            norm_refs = {str(a).strip().lower() for a in (ref.get("answers") or [])}
            correct = bool(norm_pred and norm_pred in norm_refs)

        elif task in ("ocr_vqa", "docvqa_cn"):
            ocr_pred.append(answer)
            ans_list = list(ref.get("answers") or [])
            ref_ans = ans_list[0] if ans_list else ""
            ocr_ref.append(ref_ans)
            ocr_pred_by_ds[dataset].append(answer)
            ocr_ref_by_ds[dataset].append(ref_ans)

            # 简化：预测文本是参考答案的子串即认为正确
            correct = bool(answer.strip() and answer.strip() in ref_ans)

            # OCR 证据命中率：若参考 OCR tokens 中有包含答案的 token，
            # 则要求预测 evidence 中至少有一个 OCR 类型的 id 命中这些 token。
            ref_evi = ref.get("evidence") or {}
            ref_tokens = ref_evi.get("ocr_tokens") or []
            gold_ids = set()
            if ref_ans and ref_tokens:
                for t in ref_tokens:
                    t_text = str(t.get("text") or "")
                    if t_text and (t_text in ref_ans or ref_ans in t_text):
                        tid = t.get("id")
                        if tid is not None:
                            gold_ids.add(tid)
            if gold_ids:
                ocr_evi_total += 1
                pred_ids = {
                    e.get("id")
                    for e in pred_evidence
                    if str(e.get("type") or "").startswith("ocr")
                }
                if any(pid in gold_ids for pid in pred_ids):
                    ocr_evi_hits += 1

        elif task == "chart_qa":
            try:
                pred_val = float(answer)
                true_list = list(ref.get("answers") or [])
                true_val = float(true_list[0]) if true_list else 0.0
            except ValueError:
                continue

            chart_pred_vals.append(pred_val)
            chart_ref_vals.append(true_val)
            chart_pred_by_ds[dataset].append(pred_val)
            chart_ref_by_ds[dataset].append(true_val)

            # 认为数值精确匹配时正确
            correct = (pred_val == true_val)

            # 图表元素覆盖率：从参考 chart_elements 中选择与答案最接近的元素，
            # 要求预测 evidence 中至少有一个 chart_elem 类型的 id 命中该元素。
            ref_evi = ref.get("evidence") or {}
            ref_elems = ref_evi.get("chart_elements") or []
            gold_elem_id = None
            if ref_elems:
                # 数值答案：选择数值最接近的元素
                best_idx = None
                best_diff = float("inf")
                for idx_elem, elem in enumerate(ref_elems):
                    meta = elem.get("meta") or {}
                    v = meta.get("value") or meta.get("y") or meta.get("val") or meta.get("v")
                    try:
                        v_float = float(v)
                    except Exception:  # noqa: BLE001
                        continue
                    diff = abs(v_float - true_val)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = idx_elem
                if best_idx is not None:
                    gold_elem_id = ref_elems[best_idx].get("id")

            if gold_elem_id is not None:
                chart_evi_total += 1
                pred_ids = {
                    e.get("id")
                    for e in pred_evidence
                    if str(e.get("type") or "").startswith("chart")
                }
                if gold_elem_id in pred_ids:
                    chart_evi_hits += 1

        # 用于 coverage-risk：仅统计带标注的样本
        if task in ("vqa_cn", "ocr_vqa", "docvqa_cn", "chart_qa"):
            coverage_items.append((conf, abstain, correct))

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
    coverages: List[float] = []
    risks: List[float] = []
    if coverage_items:
        # thresholds 从高到低，覆盖率随阈值降低而增大
        thresholds = [i / 20.0 for i in range(0, 21)]
        total = len(coverage_items)
        for t in thresholds:
            # 选择：未拒答且置信度 >= t
            selected = [
                item for item in coverage_items if (not item[1]) and item[0] >= t  # (conf, abstain, correct)
            ]
            if not selected:
                coverages.append(0.0)
                risks.append(0.0)
                continue
            coverage = len(selected) / total
            err = sum(1 for _, _, correct in selected if not correct) / len(selected)
            coverages.append(coverage)
            risks.append(err)
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
        # OCR 证据命中率（预测 evidence 是否命中包含答案的 OCR token）
        ocr_evi_rate = ocr_evi_hits / ocr_evi_total if ocr_evi_total > 0 else 0.0
        f.write(f"ocr_vqa,evidence_hit_rate,{ocr_evi_rate:.4f}\n")

        f.write(f"chart_qa,em,{chart_em:.4f}\n")
        f.write(f"chart_qa,mape,{chart_mape:.4f}\n")
        # 图表元素覆盖率（预测 evidence 是否命中与答案对应的图元）
        chart_evi_rate = chart_evi_hits / chart_evi_total if chart_evi_total > 0 else 0.0
        f.write(f"chart_qa,element_coverage,{chart_evi_rate:.4f}\n")

        f.write(f"answerability,ap,{ans_ap:.4f}\n")

        # 逐数据集指标
        for ds, preds_ds in vqa_pred_by_ds.items():
            refs_ds = vqa_ref_by_ds[ds]
            acc_ds = vqa_accuracy(preds_ds, refs_ds) if preds_ds else 0.0
            f.write(f"vqa_cn[{ds}],acc,{acc_ds:.4f}\n")

        for ds, preds_ds in ocr_pred_by_ds.items():
            refs_ds = ocr_ref_by_ds[ds]
            hit_ds, ed_ds = ocr_vqa_scores(preds_ds, refs_ds) if preds_ds else (0.0, 0.0)
            f.write(f"ocr_vqa[{ds}],hit_rate,{hit_ds:.4f}\n")
            f.write(f"ocr_vqa[{ds}],avg_edit_distance,{ed_ds:.4f}\n")

        for ds, preds_ds in chart_pred_by_ds.items():
            refs_ds = chart_ref_by_ds[ds]
            em_ds, mape_ds = chart_scores(preds_ds, refs_ds) if preds_ds else (0.0, 0.0)
            f.write(f"chart_qa[{ds}],em,{em_ds:.4f}\n")
            f.write(f"chart_qa[{ds}],mape,{mape_ds:.4f}\n")

    logger.info("评测完成，结果已写入: %s 与 %s", csv_path, png_path)


if __name__ == "__main__":
    main()
