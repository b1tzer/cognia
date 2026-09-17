#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Cognia 本地开发 Postgres 一键启动脚本
#
# 用途：开发调试期用本地 pg 替代远端 Supabase，把会话管理接口的 DB RTT
#       从 200-400ms 降到 <1ms。连接串与 .env 中 LANGGRAPH_DATABASE_URL 一致。
# 数据持久化在 named volume cognia-pg-data，删除容器不丢数据。
#
# 用法：
#   scripts/start-local-pg.sh          启动（已存在则直接启动，幂等）
#   scripts/start-local-pg.sh down     停止并删除容器（数据 volume 保留）
#   scripts/start-local-pg.sh reset    停止+删容器+删数据 volume（彻底清空）
# =============================================================================

CONTAINER_NAME="cognia-pg"
VOLUME_NAME="cognia-pg-data"
PORT="5432"
POSTGRES_USER="postgres"
POSTGRES_PASSWORD="password"
POSTGRES_DB="cognia"
IMAGE="postgres:16-alpine"

# 与 .env 中 LANGGRAPH_DATABASE_URL 保持一致（用 127.0.0.1 避免 IPv6 localhost 解析问题）
DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:${PORT}/${POSTGRES_DB}"

case "${1:-up}" in
  down)
    echo "[cognia-pg] 停止并删除容器（数据 volume ${VOLUME_NAME} 保留）..."
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    echo "[cognia-pg] 已停止。"
    ;;
  reset)
    echo "[cognia-pg] 停止容器并删除数据 volume（彻底清空，谨慎）..."
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker volume rm "$VOLUME_NAME" >/dev/null 2>&1 || true
    echo "[cognia-pg] 已清空。可再次运行 scripts/start-local-pg.sh 重建。"
    ;;
  up|*)
    if docker ps -a --filter "name=${CONTAINER_NAME}" --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      echo "[cognia-pg] 容器已存在，直接启动..."
      docker start "$CONTAINER_NAME" >/dev/null
    else
      echo "[cognia-pg] 创建并启动容器..."
      docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        -e POSTGRES_USER="$POSTGRES_USER" \
        -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
        -e POSTGRES_DB="$POSTGRES_DB" \
        -p "127.0.0.1:${PORT}:5432" \
        -v "${VOLUME_NAME}:/var/lib/postgresql/data" \
        "$IMAGE" >/dev/null
    fi

    echo "[cognia-pg] 等待 pg 就绪..."
    for i in $(seq 1 30); do
      if docker exec "$CONTAINER_NAME" pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
        echo "[cognia-pg] pg 已就绪（第 ${i}s）。"
        break
      fi
      sleep 1
    done

    echo "[cognia-pg] 连接串（写入 .env）："
    echo "  LANGGRAPH_DATABASE_URL=${DATABASE_URL}"
    docker ps --filter "name=${CONTAINER_NAME}" --format '  {{.Names}}\t{{.Status}}\t{{.Ports}}'
    ;;
esac
