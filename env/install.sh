#!/usr/bin/env bash

# 简易安装脚本（macOS + Apple Silicon + MPS）
#
# 功能：
#   - 创建并激活一个 Python 虚拟环境（可选）；
#   - 安装 PyTorch（含 MPS 支持）以及 OCR / 可视化等依赖；
#
# 使用示例（在仓库根目录）：
#   bash env/install.sh

set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[安装] 使用 Python: ${PYTHON_BIN}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "[安装] 创建虚拟环境: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "[安装] 激活虚拟环境"
source "${VENV_DIR}/bin/activate"

echo "[安装] 升级 pip"
pip install --upgrade pip

echo "[安装] 安装 PyTorch (含 MPS 支持) 与基础依赖"
pip install \
  torch torchvision torchaudio \
  opencv-python \
  pytesseract \
  pyyaml \
  scikit-learn \
  matplotlib \
  pillow

echo "[安装] 完成。请使用 'source ${VENV_DIR}/bin/activate' 激活环境。"
