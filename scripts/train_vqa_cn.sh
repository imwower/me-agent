#!/usr/bin/env bash

set -e

CONFIG="configs/train_vqa_cn.yaml"

echo "[训练] VQA 中文任务，配置: ${CONFIG}"
python -m src.training.trainer --config "${CONFIG}"

