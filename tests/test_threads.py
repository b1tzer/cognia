"""cognia 会话标题（自动标题）逻辑单元测试。

覆盖本次 PR 修复的四类问题（对应 Sourcery review 的 issue）：
- issue 4：AI 标题截断到 15 字（与 prompt 声明一致），降级标题截断到 30 字。
- issue 1：自动标题 SQL 为纯 UPDATE，删除线程后不会被 UPSERT 复活。
- issue 2：get_thread_title 提供「是否已有标题」判断，供后台任务跳过 LLM 调用。
- issue 7：auto_title_if_empty 只负责「空则写、非空则保留」，参数顺序正确。

全部使用 fake pool / fake 模型离线运行，不依赖 PostgreSQL 与真实 LLM。
"""

import asyncio

import pytest

from cognia import threads
from cognia.routers import threads as threads_router


# ---- fake pool（与 test_feedback.py 同款，最小够用）----

class _FakeCursor:
    """记录 execute 参数、返回预设 fetchone 结果的假游标。"""

    def __init__(self, row=None):
        self._row = row
        self.executed = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def execute(self, sql, params=None):
        self.executed = (sql, params)

    async def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def cursor(self):
        return self._cursor


class _FakePool:
    """连接池持有同一游标，测试可经 pool.cursor 访问 execute 记录。"""

    def __init__(self, row=None):
        self.cursor = _FakeCursor(row)

    def connection(self):
        return _FakeConn(self.cursor)


# ---- issue 4：标题长度上限 ----

def test_fallback_title_truncates_to_30():
    """降级标题：超过 30 字截断并加省略号。"""
    text = "啊" * 40
    assert threads_router._fallback_title(text) == "啊" * 30 + "…"


def test_fallback_title_short_no_ellipsis():
    """降级标题：不超过 30 字原样返回，不加省略号。"""
    assert threads_router._fallback_title("短标题") == "短标题"


class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeTitleModel:
    def __init__(self, content):
        self._content = content

    async def ainvoke(self, messages):
        return _FakeResp(self._content)


def test_generate_ai_title_truncates_to_15(monkeypatch):
    """issue 4：模型返回超长标题时截断到 15 字，与 prompt 声明一致。"""
    long_title = "一二三四五六七八九十一二三四五六七八九十"  # 20 字
    monkeypatch.setattr(
        threads_router.models,
        "get_planner_model",
        lambda: _FakeTitleModel(long_title),
    )
    title = asyncio.run(threads_router._generate_ai_title("任意首条消息"))
    assert title == "一二三四五六七八九十一二三四五" + "…"


def test_generate_ai_title_empty_raises(monkeypatch):
    """模型返回空内容时抛 ValueError，由调用方降级为截断标题。"""
    monkeypatch.setattr(
        threads_router.models,
        "get_planner_model",
        lambda: _FakeTitleModel("   "),
    )
    with pytest.raises(ValueError):
        asyncio.run(threads_router._generate_ai_title("任意首条消息"))


# ---- issue 1：自动标题 SQL 为纯 UPDATE ----

def test_auto_title_sql_is_update_not_insert():
    """issue 1：自动标题 SQL 必须是 UPDATE，不得含 INSERT（防复活已删除线程）。"""
    assert "UPDATE" in threads._AUTO_TITLE_SQL.upper()
    assert "INSERT" not in threads._AUTO_TITLE_SQL.upper()
    # 防覆盖护栏仍保留：只在 title 为空时更新
    assert "title IS NULL" in threads._AUTO_TITLE_SQL
    assert "title = ''" in threads._AUTO_TITLE_SQL


# ---- issue 2：get_thread_title 提供「是否已有标题」判断 ----

def test_get_thread_title_returns_existing():
    """已有标题时返回标题原文。"""
    pool = _FakePool({"title": "Spring AOP"})
    assert asyncio.run(threads.get_thread_title(pool, "t1")) == "Spring AOP"


def test_get_thread_title_empty_string_is_none():
    """空字符串标题视为「无标题」，返回 None。"""
    pool = _FakePool({"title": ""})
    assert asyncio.run(threads.get_thread_title(pool, "t1")) is None


def test_get_thread_title_missing_row_is_none():
    """线程行不存在（已删除）返回 None。"""
    pool = _FakePool(None)
    assert asyncio.run(threads.get_thread_title(pool, "t1")) is None


# ---- issue 7：auto_title_if_empty 只负责「空则写」，参数顺序正确 ----

def test_auto_title_if_empty_uses_update_param_order():
    """UPDATE 语句参数顺序必须是 (title, thread_id)。"""
    pool = _FakePool()
    asyncio.run(threads.auto_title_if_empty(pool, "t1", "新标题"))
    sql, params = pool.cursor.executed
    assert "UPDATE" in sql.upper()
    assert params == ("新标题", "t1")