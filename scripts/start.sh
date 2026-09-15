#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Cognia 一键启动脚本
#
# 架构：
#   Next.js 前端 (3000) ── /api/copilotkit ──> AG-UI 后端 (8123, cognia/server.py)
#
# 用法：
#   scripts/start.sh            默认：AG-UI 后端(后台) + Next.js 前端(前台, 3000)
#   scripts/start.sh frontend   仅启动 Next.js 前端（3000）
#   scripts/start.sh agui       仅启动 AG-UI 后端（8123）
#   scripts/start.sh chainlit   仅启动 Chainlit UI（旧教学界面）
#
# 启动前会自动检查同名服务是否已有进程在运行，有则先 kill 掉，避免端口冲突
# 或重复实例抢占资源。
# =============================================================================

# 切换到项目根目录（脚本位于 scripts/ 子目录）
cd "$(dirname "$0")/.."

mkdir -p logs

# -----------------------------------------------------------------------------
# kill_duplicates <pattern> <label>
#   pattern : pgrep -f 的进程匹配模式（完整命令行子串）
#   label   : 服务名，仅用于日志输出
# 先 SIGTERM 优雅退出，等待 1s 后仍存活则 SIGKILL 强杀。
# -----------------------------------------------------------------------------
kill_duplicates() {
    local pattern="$1"
    local label="$2"
    local pids

    pids=$(pgrep -f "$pattern" 2>/dev/null || true)

    if [ -z "$pids" ]; then
        echo "[Cognia] 未发现 ${label} 进程，无需清理。"
        return 0
    fi

    echo "[Cognia] 检测到 ${label} 已在运行（PID: $(echo "$pids" | tr '\n' ' ')），先 kill 掉..."
    echo "$pids" | xargs -r kill 2>/dev/null || true
    sleep 1

    # 仍存活则强制清理
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "[Cognia] 进程未退出，强制 kill -9（PID: $(echo "$pids" | tr '\n' ' ')）..."
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
        sleep 1
    fi
    echo "[Cognia] ${label} 旧进程已清理。"
}

SERVICE="${1:-all}"

start_agui_background() {
    echo "[Cognia] 启动 AG-UI 后端（后台, 端口 ${AGUI_PORT:-8123}）..."
    (uv run python -m uvicorn cognia.server:app \
        --host 0.0.0.0 --port "${AGUI_PORT:-8123}" --reload \
        > logs/agui.log 2>&1 &)
}

start_frontend_foreground() {
    echo "[Cognia] 启动 Next.js 前端（端口 ${FRONTEND_PORT:-3000}）..."
    cd frontend
    exec pnpm exec next dev --port "${FRONTEND_PORT:-3000}" 2>&1 | tee -a ../logs/frontend.log
}

case "$SERVICE" in
    all)
        kill_duplicates "uvicorn cognia.server:app" "AG-UI 后端"
        kill_duplicates "next dev" "Next.js 前端"
        start_agui_background
        start_frontend_foreground
        ;;
    frontend)
        kill_duplicates "next dev" "Next.js 前端"
        start_frontend_foreground
        ;;
    agui)
        kill_duplicates "uvicorn cognia.server:app" "AG-UI 后端"
        echo "[Cognia] 启动 AG-UI 后端 ..."
        exec uv run python -m uvicorn cognia.server:app \
            --host 0.0.0.0 --port "${AGUI_PORT:-8123}" --reload
        ;;
    chainlit)
        kill_duplicates "chainlit run cognia/app.py" "Chainlit UI"
        echo "[Cognia] 启动 Chainlit UI ..."
        exec uv run chainlit run cognia/app.py 2>&1 | tee -a logs/cognia.log
        ;;
    *)
        echo "用法: $0 [all|frontend|agui|chainlit]" >&2
        exit 1
        ;;
esac
