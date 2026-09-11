#!/usr/bin/env bash
# Cognia 一键启动脚本
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/backend"

# 初始化后端虚拟环境（若不存在）
if [ ! -x ".venv/bin/uvicorn" ]; then
  echo "[Cognia] 初始化后端虚拟环境..."
  PY=""
  # 优先选择兼容 pydantic-core 的 3.11/3.12/3.13（避免 3.14 的 PyO3 兼容问题）
  for c in python3.12 python3.11 python3.13 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
  # 兜底：mise 管理的 Python
  if [ -z "$PY" ] && [ -d "$HOME/.local/share/mise/installs/python" ]; then
    for v in 3.12.13 3.11.0; do
      if [ -x "$HOME/.local/share/mise/installs/python/$v/bin/python3" ]; then
        PY="$HOME/.local/share/mise/installs/python/$v/bin/python3"; break
      fi
    done
  fi
  if [ -z "$PY" ]; then echo "[Cognia] 未找到可用的 Python，请先安装 Python 3.11~3.13"; exit 1; fi
  "$PY" -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
  echo "[Cognia] 后端依赖就绪"
fi

# 构建前端（若未构建）
if [ ! -d "$ROOT/frontend/dist" ]; then
  echo "[Cognia] 构建前端..."
  (cd "$ROOT/frontend" && pnpm install && pnpm build)
  echo "[Cognia] 前端构建完成"
fi

HOST="${COGNIA_HOST:-0.0.0.0}"
PORT="${COGNIA_PORT:-8000}"
echo "[Cognia] 启动服务：http://localhost:$PORT"
exec .venv/bin/uvicorn main:app --host "$HOST" --port "$PORT"
