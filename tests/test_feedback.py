"""cognia.feedback 消息反馈数据层单元测试。

feedback 的 SQL 层依赖 psycopg AsyncConnectionPool，与 threads.py 一致——
SQL 本体不在此单测（需真 PG），这里只验证不依赖 IO 的纯逻辑：
1. upsert_feedback 对非法 feedback 值抛 ValueError（在 SQL 执行前）。
2. list_feedback 把查询行正确组装为 {message_id: feedback} 映射。

全部使用 fake pool 离线运行，不依赖 PostgreSQL。
"""

import asyncio

import pytest

from cognia import feedback


class _FakeCursor:
    """记录 execute 参数、返回预设 fetchall 结果的假游标。"""

    def __init__(self, rows):
        self._rows = rows
        self.executed = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def execute(self, sql, params=None):
        self.executed = (sql, params)

    async def fetchall(self):
        return self._rows

    async def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    """假连接：cursor() 返回假游标。"""

    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def cursor(self):
        return _FakeCursor(self._rows)


class _FakePool:
    """假连接池：connection() 返回假连接。"""

    def __init__(self, rows):
        self._rows = rows

    def connection(self):
        return _FakeConn(self._rows)


def test_upsert_feedback_rejects_invalid_value():
    """非法 feedback 值抛 ValueError（不落库）。"""
    with pytest.raises(ValueError):
        asyncio.run(
            feedback.upsert_feedback(None, "t1", "m1", "u1", "bad")
        )


def test_list_feedback_assembles_map():
    """list_feedback 把查询行组装为 {message_id: feedback} 映射。"""
    pool = _FakePool([
        {"message_id": "m1", "feedback": "up"},
        {"message_id": "m2", "feedback": "down"},
    ])
    result = asyncio.run(feedback.list_feedback(pool, "t1", "u1"))
    assert result == {"m1": "up", "m2": "down"}


def test_list_feedback_empty():
    """无反馈时返回空映射。"""
    pool = _FakePool([])
    result = asyncio.run(feedback.list_feedback(pool, "t1", "u1"))
    assert result == {}


def test_upsert_feedback_returns_row():
    """upsert_feedback 返回落库后的记录（含 feedback 字段）。"""
    pool = _FakePool([
        {
            "thread_id": "t1",
            "message_id": "m1",
            "user_id": "u1",
            "feedback": "up",
            "updated_at": "2026-09-22T00:00:00+00:00",
        }
    ])
    result = asyncio.run(
        feedback.upsert_feedback(pool, "t1", "m1", "u1", "up")
    )
    assert result["feedback"] == "up"
    assert result["message_id"] == "m1"
