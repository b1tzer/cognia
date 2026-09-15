"""cognia.app Chainlit UI 接线单元测试。

聚焦可脱离 Chainlit 运行时的纯函数（_build_config / _persist_deltas），
验证任务⑦最关键的两条接线：
1. runtime config 正确注入 thread_id + 匿名 user_id
2. graph 产出的 proficiency_deltas 能写入长期 Store（graph → Store 闭环）
"""

import asyncio

from langgraph.store.memory import InMemoryStore

from cognia.app import _build_config, _persist_deltas
from cognia.memory import get_current_proficiency, get_proficiency_history
from cognia.schemas import CognitiveState, ProficiencyEntry


def test_build_config_injects_thread_and_user_id():
    """config 同时携带 thread_id（会话恢复）与 user_id（Store 隔离）。"""
    config = _build_config("thread-1", "anon-123")
    assert config["configurable"]["thread_id"] == "thread-1"
    assert config["configurable"]["user_id"] == "anon-123"


def test_persist_deltas_writes_to_store():
    """graph 产出的 Delta dict 能写入 Store 并被读回（graph → Store 闭环）。"""
    store = InMemoryStore()
    entry = ProficiencyEntry(
        point_id="aop-concept",
        from_state=None,
        to_state=CognitiveState.PARTIAL,
        evidence=["用户说 AOP 就是切面"],
    )
    state = {"proficiency_deltas": [entry.model_dump(mode="json")]}

    asyncio.run(_persist_deltas(store, "u1", state))

    assert get_current_proficiency(store, "u1", "aop-concept") == "partial"


def test_persist_deltas_is_idempotent():
    """重复写入同一条 Delta 不产生重复条目（key = point_id:timestamp 唯一）。"""
    store = InMemoryStore()
    entry = ProficiencyEntry(
        point_id="aop-concept",
        from_state=None,
        to_state=CognitiveState.MASTERED,
        evidence=["完整解释"],
    )
    state = {"proficiency_deltas": [entry.model_dump(mode="json")]}

    asyncio.run(_persist_deltas(store, "u1", state))
    asyncio.run(_persist_deltas(store, "u1", state))

    history = get_proficiency_history(store, "u1", "aop-concept")
    assert len(history) == 1  # 幂等，不重复


def test_persist_deltas_empty_state_noop():
    """无 Delta 时写入为 noop（不抛错、不产生记录）。"""
    store = InMemoryStore()
    asyncio.run(_persist_deltas(store, "u1", {}))
    asyncio.run(_persist_deltas(store, "u1", {"proficiency_deltas": []}))
    assert get_current_proficiency(store, "u1", "any") is None
