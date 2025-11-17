#!/usr/bin/env bash

set -e

PRED_FILE="${1:-outputs/predictions.jsonl}"
REF_FILE="${2:-outputs/references.jsonl}"

echo "[评测] 使用预测文件: ${PRED_FILE}"
echo "[评测] 使用参考文件: ${REF_FILE}"

python -m src.eval.run_eval_all --pred "${PRED_FILE}" --ref "${REF_FILE}"
