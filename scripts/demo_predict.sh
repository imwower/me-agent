#!/usr/bin/env bash

set -e

if [ "$#" -lt 2 ]; then
  echo "用法: bash scripts/demo_predict.sh <image_path> <question>"
  exit 1
fi

IMAGE_PATH="$1"
QUESTION="$2"

echo "[Demo] 对图像进行中文多模态问答："
echo "  图像: ${IMAGE_PATH}"
echo "  问题: ${QUESTION}"

python -m src.inference.predictor "${IMAGE_PATH}" "${QUESTION}"

