#!/usr/bin/env bash
set -euo pipefail

# 切换到项目根目录（脚本位于 scripts/ 子目录）
cd "$(dirname "$0")/.."

# 启动 Cognia AG-UI 后端服务（CopilotKit 集成）
# 监听 0.0.0.0:8123，AG-UI 端点路径 /
exec uv run python -m uvicorn cognia.server:app --host 0.0.0.0 --port "${AGUI_PORT:-8123}" --reload
