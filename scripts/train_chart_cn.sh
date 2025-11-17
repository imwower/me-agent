#!/usr/bin/env bash

set -e

CONFIG="configs/train_chart_cn.yaml"

echo "[训练] 图表问答/值抽取 中文任务，配置: ${CONFIG}"
python -m src.training.trainer --config "${CONFIG}"

