#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# 使用示例 workspace（已写入 self-snn 路径），启用 brain-mode 跑一次 devloop
python scripts/run_orchestrator.py \
  --workspace configs/workspace.example.json \
  --mode devloop \
  --use-brain \
  --scenarios self_intro
