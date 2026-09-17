#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Cognia 远端 Supabase → 本地 Postgres 数据同步脚本
#
# 把远端托管 pg 的 5 张业务表（checkpoints / checkpoint_blobs /
# checkpoint_writes / store / cognia_threads）数据同步到本地 cognia-pg 容器。
# 不动 migrations 迁移记录表（LangGraph 依赖它们判断是否已迁移）。
#
# 为什么用 postgres:17 临时容器做 pg_dump：
#   远端 Supabase 是 PostgreSQL 17.x，本地容器是 16.x；pg_dump 版本必须
#   >= 服务端版本，16 的 pg_dump 会报 server version mismatch。故临时起
#   postgres:17-alpine 容器执行 dump，dump 出的标准 COPY SQL 可回灌 16。
#
# 用法：
#   export REMOTE_DB_URL='postgresql://user:pass@host:5432/db?sslmode=require'
#   scripts/sync-remote-to-local.sh
#   或
#   scripts/sync-remote-to-local.sh --remote-url 'postgresql://user:pass@host:5432/db?sslmode=require'
#
# 依赖：docker（本地 cognia-pg 容器需已启动，见 scripts/start-local-pg.sh）
# =============================================================================

LOCAL_CONTAINER="cognia-pg"
LOCAL_DB="cognia"
LOCAL_USER="postgres"
DUMP_IMAGE="postgres:17-alpine"
DUMP_FILE="/tmp/cognia_remote_data.sql"
# 业务表（不含 *_migrations 迁移记录表）
TABLES=(checkpoints checkpoint_blobs checkpoint_writes store cognia_threads)

REMOTE_URL="${REMOTE_DB_URL:-}"

# 解析 --remote-url 参数（可选，优先于 REMOTE_DB_URL 环境变量）
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-url)
      REMOTE_URL="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      echo "未知参数：$1（支持 --remote-url / -h）" >&2
      exit 1
      ;;
  esac
done

if [ -z "$REMOTE_URL" ]; then
  echo "错误：缺少远端连接串。" >&2
  echo "  方式一：export REMOTE_DB_URL='postgresql://user:pass@host:5432/db?sslmode=require'" >&2
  echo "  方式二：$0 --remote-url 'postgresql://user:pass@host:5432/db?sslmode=require'" >&2
  exit 1
fi

if ! docker ps --filter "name=${LOCAL_CONTAINER}" --format '{{.Names}}' | grep -qx "$LOCAL_CONTAINER"; then
  echo "错误：本地容器 ${LOCAL_CONTAINER} 未运行，请先执行 scripts/start-local-pg.sh" >&2
  exit 1
fi

# 把表清单拼成 pg_dump 的 -t 参数
DUMP_TABLE_ARGS=()
for t in "${TABLES[@]}"; do
  DUMP_TABLE_ARGS+=(-t "$t")
done

# 1) dump（临时 postgres:17 容器，用完即删）
echo "[sync] 从远端 dump 5 张业务表..."
docker run --rm "$DUMP_IMAGE" pg_dump "$REMOTE_URL" \
  --data-only --no-owner --no-privileges \
  "${DUMP_TABLE_ARGS[@]}" > "$DUMP_FILE"
echo "[sync] dump 完成：$(du -h "$DUMP_FILE" | cut -f1)"

# 2) 清空本地业务表（只清业务表，不动 migrations）
echo "[sync] 清空本地业务表..."
docker exec "$LOCAL_CONTAINER" psql -U "$LOCAL_USER" -d "$LOCAL_DB" \
  -c "TRUNCATE ${TABLES[*]};" >/dev/null

# 3) 恢复
echo "[sync] 恢复到本地..."
cat "$DUMP_FILE" | docker exec -i "$LOCAL_CONTAINER" psql -U "$LOCAL_USER" -d "$LOCAL_DB" >/dev/null

# 4) 验证行数一致
echo "[sync] 验证行数（表 | 本地 | 远端）:"
FAIL=0
for t in "${TABLES[@]}"; do
  local_cnt=$(docker exec "$LOCAL_CONTAINER" psql -U "$LOCAL_USER" -d "$LOCAL_DB" -Atc "SELECT count(*) FROM $t;" 2>/dev/null)
  remote_cnt=$(docker exec "$LOCAL_CONTAINER" psql "$REMOTE_URL" -Atc "SELECT count(*) FROM $t;" 2>/dev/null)
  mark="❌"; [ "$local_cnt" = "$remote_cnt" ] && mark="✅"
  echo "  $t | $local_cnt | $remote_cnt $mark"
  [ "$local_cnt" != "$remote_cnt" ] && FAIL=1
done

if [ "$FAIL" -eq 0 ]; then
  echo "[sync] 全部一致，同步完成。"
else
  echo "[sync] 存在不一致，请检查。" >&2
  exit 1
fi
