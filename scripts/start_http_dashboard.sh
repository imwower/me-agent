#!/usr/bin/env bash
# 启动 HTTP Dashboard，提供 /status /experiments/recent /notebook/recent 等接口，并托管静态页面。
set -e
cd "$(dirname "$0")/.."

python - <<'PY'
import sys
import time
from me_core.world_model import SimpleWorldModel
from me_core.self_model import SimpleSelfModel
from me_ext.http_api import serve_http

serve_http(SimpleWorldModel(), SimpleSelfModel(), port=8000)
print("HTTP server on 8000. 按 Ctrl+C 停止，或在非交互模式下直接 kill 进程。")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    sys.exit(0)
PY
