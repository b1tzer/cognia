"""Cognia 线程元数据层（会话列表 / title 持久化）。

权威会话数据在 LangGraph checkpointer（PostgreSQL `checkpoints` 表，按
`thread_id` 区分会话，checkpoint JSONB 内含 `ts` 时间戳）。但 checkpoints
表是「checkpoint 级别」的——一个 thread 下有多条 checkpoint 记录，天然
不适合存 thread 级别的 title / created_at 元数据（这正是「checkpoint 本身
不适合可靠保存 title」的判断依据）。

因此本模块在同一个 PostgreSQL 里维护一张极简的 `cognia_threads` 表，只存
thread_id + title + created_at + updated_at 四列，与 checkpointer 复用同一个
AsyncConnectionPool（见 memory.get_pool）。不引入任何新数据库、新框架或
第三方组件。

会话列表（list_threads）以 checkpoints 表为「真实活动」来源：
- thread_id 集合 = cognia_threads ∪ checkpoints 的并集（兼容升级前只有
  checkpoint、没有 cognia_threads 记录的旧会话）；
- updated_at 优先取 checkpoint 最新 ts（真实对话活动时间），空会话（尚未
  产生 checkpoint）则回退到 cognia_threads.updated_at。
"""

from __future__ import annotations

import uuid
from typing import Any

# ---- SQL（PostgreSQL；未加引号的标识符自动转小写） ----

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cognia_threads (
    thread_id   TEXT PRIMARY KEY,
    title       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# 以 cognia_threads 为主表，FULL OUTER JOIN checkpoints 聚合子查询，
# 保证「升级前只有 checkpoint 的旧会话」也能被列出；updated_at 优先真实对话时间。
_LIST_THREADS_SQL = """
SELECT
    COALESCE(t.thread_id, c.thread_id) AS thread_id,
    t.title,
    COALESCE(t.created_at, (c.first_ts)::timestamptz) AS created_at,
    COALESCE((c.latest_ts)::timestamptz, t.updated_at) AS updated_at
FROM cognia_threads t
FULL OUTER JOIN (
    SELECT
        thread_id,
        MIN(checkpoint->>'ts') AS first_ts,
        MAX(checkpoint->>'ts') AS latest_ts
    FROM checkpoints
    WHERE checkpoint_ns = ''
    GROUP BY thread_id
) c ON c.thread_id = t.thread_id
ORDER BY COALESCE((c.latest_ts)::timestamptz, t.updated_at) DESC
"""

_CREATE_THREAD_SQL = """
INSERT INTO cognia_threads (thread_id, title)
VALUES (%s, %s)
RETURNING thread_id, title, created_at, updated_at
"""

# 用 UPSERT 兼容「升级前只有 checkpoint、尚无 cognia_threads 记录」的旧会话：
# 首次重命名时补建一条元数据记录；已存在的记录则更新 title 与 updated_at。
_RENAME_THREAD_SQL = """
INSERT INTO cognia_threads (thread_id, title, created_at, updated_at)
VALUES (%s, %s, now(), now())
ON CONFLICT (thread_id) DO UPDATE SET title = EXCLUDED.title, updated_at = now()
RETURNING thread_id, title, created_at, updated_at
"""

_DELETE_THREAD_META_SQL = "DELETE FROM cognia_threads WHERE thread_id = %s"

# 自动标题写入：纯 UPDATE，仅更新「已存在且 title 为空」的行。
# 相比旧 UPSERT，这里有两个关键保证：
# 1. 防覆盖：WHERE title IS NULL OR title = '' 保证用户手动重命名后不被改写。
# 2. 防复活：线程被删除（cognia_threads 行已 DELETE）后，后台标题任务再执行
#    UPDATE 时无行可匹配、不插入新行，被删除的会话不会凭空回到列表。
# 副作用：升级前只有 checkpoint、无 cognia_threads 行的旧会话，首次自动标题
# 不会补建元数据（这类会话标题为空，仍可在前端 rename 时经 _RENAME_THREAD_SQL
# 的 UPSERT 补建，不影响手动命名路径）。
_AUTO_TITLE_SQL = """
UPDATE cognia_threads
SET title = %s, updated_at = now()
WHERE thread_id = %s
  AND (title IS NULL OR title = '')
"""

_GET_THREAD_TITLE_SQL = "SELECT title FROM cognia_threads WHERE thread_id = %s"


def _iso(value: Any) -> str | None:
    """把 psycopg 返回的 datetime / None / str 统一为 ISO 字符串。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def _row_to_thread(row: dict) -> dict:
    """把查询行转为对外的 thread 字典。"""
    return {
        "thread_id": row["thread_id"],
        "title": row["title"],
        "created_at": _iso(row["created_at"]),
        "updated_at": _iso(row["updated_at"]),
    }


async def ensure_schema(pool) -> None:
    """建 `cognia_threads` 表（幂等）。复用调用方传入的连接池。"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_SCHEMA_SQL)


async def list_threads(pool) -> list[dict]:
    """列出全部会话，按 updated_at 降序（最新会话在前）。"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_LIST_THREADS_SQL)
            rows = await cur.fetchall()
    return [_row_to_thread(r) for r in rows]


async def create_thread(pool, thread_id: str | None = None, title: str | None = None) -> dict:
    """创建新会话元数据，返回新 thread 记录。

    thread_id 缺省时由后端生成 UUID（标准带连字符格式，与前端历史格式一致）。
    """
    thread_id = thread_id or str(uuid.uuid4())
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_CREATE_THREAD_SQL, (thread_id, title))
            row = await cur.fetchone()
    return _row_to_thread(row)


async def rename_thread(pool, thread_id: str, title: str) -> dict:
    """重命名会话（UPSERT，兼容旧会话首次补建元数据）。"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_RENAME_THREAD_SQL, (thread_id, title))
            row = await cur.fetchone()
    return _row_to_thread(row)


async def delete_thread(pool, checkpointer, thread_id: str) -> None:
    """删除会话：先删元数据，再删 LangGraph checkpoint（含 blobs / writes）。"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_DELETE_THREAD_META_SQL, (thread_id,))
    await checkpointer.adelete_thread(thread_id)


async def get_thread_title(pool, thread_id: str) -> str | None:
    """读会话标题；线程不存在或无标题（NULL/''）时返回 None。

    供自动标题任务在调用 LLM 前判断「是否已有标题」，避免对已命名会话
    重复调用模型（auto_title_if_empty 的 SQL 只防覆盖、不省模型调用）。
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_GET_THREAD_TITLE_SQL, (thread_id,))
            row = await cur.fetchone()
    if row is None:
        return None
    title = row["title"]
    return title if title else None


async def auto_title_if_empty(pool, thread_id: str, title: str) -> None:
    """仅当会话尚无 title 时，写入调用方传入的自动标题。

    写入的 title 可能是 AI 精炼的语义标题，也可能是截断首条用户消息的
    降级标题——由调用方（routers/threads._auto_title_task）决定，本函数
    不关心其来源，只负责「空则写、非空则保留」的持久化语义。

    - title 已存在（用户手动命名过）时保持不变，不会被覆盖。
    - 线程行已被删除时，纯 UPDATE 不匹配任何行、不插入新行，避免复活
      已删除会话。
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_AUTO_TITLE_SQL, (title, thread_id))
