"""Cognia 消息反馈层（点赞 / 点踩持久化）。

与线程元数据层（threads.py）同风格：在同一个 PostgreSQL 里维护一张极简
`message_feedback` 表，复用 AsyncConnectionPool，不引入任何新数据库。

语义（对齐 ChatGPT 等主流产品）：
- 同一用户对同一消息只有一条反馈（up / down / 无），UNIQUE 约束保证。
- 点赞 → upsert feedback='up'；点踩 → upsert feedback='down'。
- 再点同一种（取消）→ delete；切换（up→down / down→up）→ upsert 覆盖。
- feedback 按 message_id 关联前端气泡；重试后旧消息被截断，旧反馈成为
  孤儿数据（不影响功能，后续可按 updated_at 清理）。

匿名 user_id 由前端显式传参（与 knowledge-map 端点一致），不走 AG-UI
forwarded_props——feedback 是独立 HTTP 端点，无 AG-UI 输入上下文。
"""

from __future__ import annotations

from typing import Any

# ---- SQL（PostgreSQL；未加引号的标识符自动转小写） ----

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS message_feedback (
    id          BIGSERIAL PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    message_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    feedback    TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (thread_id, message_id, user_id)
);
"""

# UPSERT：已存在则更新 feedback 与 updated_at，实现「切换」语义。
_UPSERT_SQL = """
INSERT INTO message_feedback (thread_id, message_id, user_id, feedback, created_at, updated_at)
VALUES (%s, %s, %s, %s, now(), now())
ON CONFLICT (thread_id, message_id, user_id)
DO UPDATE SET feedback = EXCLUDED.feedback, updated_at = now()
RETURNING thread_id, message_id, user_id, feedback, updated_at
"""

_DELETE_SQL = """
DELETE FROM message_feedback
WHERE thread_id = %s AND message_id = %s AND user_id = %s
"""

# 按会话拉取全部反馈，供前端刷新后恢复高亮。UNIQUE 已保证 message_id 唯一。
_LIST_SQL = """
SELECT message_id, feedback
FROM message_feedback
WHERE thread_id = %s AND user_id = %s
"""

VALID_FEEDBACK = {"up", "down"}


def _iso(value: Any) -> str | None:
    """把 psycopg 返回的 datetime / None / str 统一为 ISO 字符串。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def _row_to_feedback(row: dict) -> dict:
    """把查询行转为对外的 feedback 字典。"""
    return {
        "thread_id": row["thread_id"],
        "message_id": row["message_id"],
        "user_id": row["user_id"],
        "feedback": row["feedback"],
        "updated_at": _iso(row["updated_at"]),
    }


async def ensure_schema(pool) -> None:
    """建 `message_feedback` 表（幂等）。复用调用方传入的连接池。"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_SCHEMA_SQL)


async def upsert_feedback(
    pool, thread_id: str, message_id: str, user_id: str, feedback: str
) -> dict:
    """写入 / 覆盖一条反馈（up / down）。返回落库后的记录。"""
    if feedback not in VALID_FEEDBACK:
        raise ValueError(f"非法 feedback: {feedback!r}，仅支持 up / down")
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                _UPSERT_SQL, (thread_id, message_id, user_id, feedback)
            )
            row = await cur.fetchone()
    return _row_to_feedback(row)


async def delete_feedback(pool, thread_id: str, message_id: str, user_id: str) -> None:
    """取消一条反馈（再次点同一种）。幂等：不存在也不报错。"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_DELETE_SQL, (thread_id, message_id, user_id))


async def list_feedback(pool, thread_id: str, user_id: str) -> dict[str, str]:
    """读某会话的全部反馈，返回 `{message_id: feedback}` 映射。"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_LIST_SQL, (thread_id, user_id))
            rows = await cur.fetchall()
    return {r["message_id"]: r["feedback"] for r in rows}
