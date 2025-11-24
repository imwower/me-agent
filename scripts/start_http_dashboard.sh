#!/usr/bin/env bash
# 启动 HTTP Dashboard，提供 /status /experiments/recent /notebook/recent 等接口，并托管静态页面。
set -e
cd "$(dirname "$0")/.."

python - <<'PY'
from me_core.world_model import SimpleWorldModel
from me_core.self_model import SimpleSelfModel
from me_ext.http_api import serve_http

serve_http(SimpleWorldModel(), SimpleSelfModel(), port=8000)
input("HTTP server on 8000, Enter to stop")
PY
