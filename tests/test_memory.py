"""cognia.memory 记忆层单元测试。

使用 InMemoryStore（不依赖 Postgres），验证：
1. 同一 user_id 跨会话读历史熟练度
2. 不同 user_id 数据隔离
3. 写库走 Delta 追加（不覆盖）
"""

from datetime import datetime, timezone

from langgraph.store.memory import InMemoryStore

from cognia.memory import (
    append_proficiency_delta,
    get_checkpointer,
    get_current_proficiency,
    get_profile,
    get_proficiency_history,
    get_store,
    get_user_id,
    put_profile,
)
from cognia.schemas import CognitiveState, ProficiencyEntry


def _entry(point_id, from_state, to_state, day):
    """构造带确定 timestamp 的 ProficiencyEntry，保证 key 唯一。"""
    return ProficiencyEntry(
        point_id=point_id,
        from_state=from_state,
        to_state=to_state,
        evidence=[f"证据-{day}"],
        timestamp=datetime(2026, 9, day, tzinfo=timezone.utc),
    )


def test_delta_append_preserves_history():
    """同一点位多次 Delta 追加，历史完整保留（不覆盖，宪法 §5）。"""
    store = InMemoryStore()
    e1 = _entry("p1", None, CognitiveState.PARTIAL, 1)
    e2 = _entry("p1", CognitiveState.PARTIAL, CognitiveState.MASTERED, 2)
    append_proficiency_delta(store, "u1", e1)
    append_proficiency_delta(store, "u1", e2)

    history = get_proficiency_history(store, "u1", "p1")
    assert len(history) == 2
    assert history[0]["to_state"] == "partial"
    assert history[1]["to_state"] == "mastered"
    assert history[0]["from_state"] is None
    assert history[1]["from_state"] == "partial"


def test_current_proficiency_is_latest_delta():
    """当前熟练度 = 最新 Delta 的 to_state。"""
    store = InMemoryStore()
    e1 = _entry("p1", None, CognitiveState.PARTIAL, 1)
    e2 = _entry("p1", CognitiveState.PARTIAL, CognitiveState.MASTERED, 2)
    append_proficiency_delta(store, "u1", e1)
    append_proficiency_delta(store, "u1", e2)

    assert get_current_proficiency(store, "u1", "p1") == "mastered"


def test_user_isolation():
    """不同 user_id 数据隔离：u1 的数据对 u2 不可见。"""
    store = InMemoryStore()
    e1 = _entry("p1", None, CognitiveState.PARTIAL, 1)
    append_proficiency_delta(store, "u1", e1)

    assert get_current_proficiency(store, "u2", "p1") is None
    assert get_proficiency_history(store, "u2", "p1") == []


def test_point_isolation():
    """同 user 不同知识点互不串扰。"""
    store = InMemoryStore()
    append_proficiency_delta(store, "u1", _entry("p1", None, CognitiveState.PARTIAL, 1))
    assert get_current_proficiency(store, "u1", "p2") is None


def test_get_user_id_from_runtime_context():
    """匿名 user_id 从 runtime context 注入（不塞 State）。"""
    config = {"configurable": {"thread_id": "t1", "user_id": "anon-123"}}
    assert get_user_id(config) == "anon-123"
    assert get_user_id({"configurable": {"thread_id": "t1"}}) is None


def test_profile_read_write():
    """画像偏好可写可读。"""
    store = InMemoryStore()
    put_profile(store, "u1", "language", {"pref": "zh", "tone": "concise"})
    assert get_profile(store, "u1", "language") == {"pref": "zh", "tone": "concise"}
    assert get_profile(store, "u1", "nonexistent") is None
    assert get_profile(store, "u2", "language") is None  # 画像同样按 user 隔离


def test_factory_pool_has_autocommit(monkeypatch):
    """工厂函数的 ConnectionPool 必须带 autocommit=True。

    这是「部署后静默丢数据」bug 的最小回归防线：langgraph 的 put() 从不显式
    commit，全靠连接的 autocommit 立即提交；一旦有人手滑删掉 kwargs 里的
    autocommit=True，写操作会在连接归还池时被回滚，checkpoint / 熟练度 Delta
    静默丢失。由于本环境无 Postgres、无法用真库验证，只能捕获构造参数断言，
    兜住「参数被删」这一层。同时覆盖 checkpointer 与 store 两个工厂。
    """
    captured = {}

    class FakePool:
        @classmethod
        def __class_getitem__(cls, item):
            # langgraph 的 _internal.py 在 import 时执行
            # `ConnectionPool[Connection[DictRow]]`，需支持下标访问（返回自身即可）。
            return cls

        def __init__(self, conninfo, **kwargs):
            captured.setdefault("kwargs_list", []).append(kwargs.get("kwargs", {}))

    class FakeSaver:
        def __init__(self, conn):
            pass

        def setup(self):
            pass

    class FakeStore:
        def __init__(self, conn):
            pass

        def setup(self):
            pass

    monkeypatch.setenv("DATABASE_URL", "postgres://x:x@localhost/x")
    monkeypatch.setattr("psycopg_pool.ConnectionPool", FakePool)
    monkeypatch.setattr("langgraph.checkpoint.postgres.PostgresSaver", FakeSaver)
    monkeypatch.setattr("langgraph.store.postgres.PostgresStore", FakeStore)

    get_checkpointer()
    get_store()

    assert len(captured["kwargs_list"]) == 2
    for kwargs in captured["kwargs_list"]:
        assert kwargs["autocommit"] is True
