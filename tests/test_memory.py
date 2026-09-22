"""cognia.memory 记忆层单元测试。

使用 InMemoryStore（不依赖 Postgres），验证：
1. 同一 user_id 跨会话读历史熟练度
2. 不同 user_id 数据隔离
3. 写库走 Delta 追加（不覆盖）
"""

import asyncio
from datetime import datetime, timezone

import pytest
from langgraph.store.memory import InMemoryStore

from cognia import memory as memory_mod
from cognia.memory import (
    add_prerequisite,
    alist_authoritative_proficiencies,
    alist_knowledge_models,
    aput_profile,
    arecord_observation,
    aquery_observations,
    aquery_proficiency,
    delete_point,
    get_checkpointer,
    get_knowledge_model,
    get_profile,
    get_profile_dict,
    get_store,
    get_user_id,
    insert_point,
    list_knowledge_models,
    normalize_goal,
    put_knowledge_model,
    put_profile,
    query_dependents,
    query_observations,
    query_point,
    query_prerequisites,
    query_proficiency,
    query_structure,
    record_observation,
    remove_prerequisite,
    set_attributes,
    update_point,
)
from cognia.schemas import (
    BloomLevel,
    CognitiveState,
    Confidence,
    KnowledgePoint,
    Observation,
    PointAttributes,
    PointType,
)


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


def test_get_profile_dict_merges_fields():
    """get_profile_dict 读取某 user 全部画像字段并拍平成 dict。"""
    store = InMemoryStore()
    put_profile(store, "u1", "language", {"value": "zh"})
    put_profile(store, "u1", "communication_style", {"value": "简洁"})
    assert get_profile_dict(store, "u1") == {
        "language": "zh",
        "communication_style": "简洁",
    }
    assert get_profile_dict(store, "u2") == {}  # 不同 user 隔离


def test_aput_profile_writes():
    """aput_profile 异步写入，读回一致（供 Chainlit 主事件循环调用）。"""
    store = InMemoryStore()
    asyncio.run(memory_mod.aput_profile(store, "u1", "language", {"value": "zh"}))
    assert get_profile(store, "u1", "language") == {"value": "zh"}


def test_factory_pool_has_autocommit(monkeypatch):
    """工厂函数的 ConnectionPool 必须带 autocommit=True。

    这是「部署后静默丢数据」bug 的最小回归防线：langgraph 的 put() 从不显式
    commit，全靠连接的 autocommit 立即提交；一旦有人手滑删掉 kwargs 里的
    autocommit=True，写操作会在连接归还池时被回滚，checkpoint / 熟练度 Delta
    静默丢失。由于本环境无 Postgres、无法用真库验证，只能捕获构造参数断言，
    兜住「参数被删」这一层。同时覆盖 checkpointer 与 store 两个异步工厂。
    """
    captured = {}

    class FakeAsyncPool:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, conninfo, **kwargs):
            captured.setdefault("kwargs_list", []).append(kwargs.get("kwargs", {}))

        async def open(self, wait=False):
            pass

    class FakeAsyncSaver:
        def __init__(self, conn):
            pass

        async def setup(self):
            pass

    class FakeAsyncStore:
        def __init__(self, conn):
            pass

        async def setup(self):
            pass

    # 重置单例缓存，避免被其他用例污染
    memory_mod._checkpointer_cache = None
    memory_mod._store_cache = None

    monkeypatch.setenv("LANGGRAPH_DATABASE_URL", "postgres://x:x@localhost/x")
    monkeypatch.setattr("psycopg_pool.AsyncConnectionPool", FakeAsyncPool)
    monkeypatch.setattr("langgraph.checkpoint.postgres.aio.AsyncPostgresSaver", FakeAsyncSaver)
    monkeypatch.setattr("langgraph.store.postgres.aio.AsyncPostgresStore", FakeAsyncStore)

    async def _init():
        await get_checkpointer()
        await get_store()

    asyncio.run(_init())

    # Checkpointer 与 Store 共享同一个 AsyncConnectionPool（_get_pool 单例），
    # 因此连接池只创建一次。
    assert len(captured["kwargs_list"]) == 1
    for kwargs in captured["kwargs_list"]:
        assert kwargs["autocommit"] is True


# ---- 知识模型持久化（Task ⑨）----

def test_normalize_goal():
    """归一化目标：去空白 + 统一小写，等价目标映射到同一 key。"""
    assert normalize_goal("Spring AOP") == "spring aop"
    assert normalize_goal("  Spring   AOP  ") == "spring aop"
    assert normalize_goal("SPRING AOP") == "spring aop"
    assert normalize_goal("spring\taop") == "spring aop"  # tab 折叠为空格


def test_knowledge_model_read_write():
    """知识模型可写可读，缺失返回 None。"""
    store = InMemoryStore()
    km = {"goal": "Spring AOP", "points": [{"id": "aop-concept", "name": "AOP 概念"}]}
    goal_key = normalize_goal("Spring AOP")

    put_knowledge_model(store, "u1", goal_key, km)
    assert get_knowledge_model(store, "u1", goal_key) == km
    assert get_knowledge_model(store, "u1", "other-goal") is None


def test_knowledge_model_user_isolation():
    """知识模型按 user 隔离。"""
    store = InMemoryStore()
    km = {"goal": "Spring AOP", "points": []}
    put_knowledge_model(store, "u1", "spring aop", km)
    assert get_knowledge_model(store, "u2", "spring aop") is None


# ---- 聚合读取（个人页 / 知识版图）----

def test_list_knowledge_models_returns_all_goals():
    """list_knowledge_models 返回某 user 全部知识模型。"""
    store = InMemoryStore()
    put_knowledge_model(store, "u1", "spring aop", {"goal": "Spring AOP", "points": []})
    put_knowledge_model(store, "u1", "react hooks", {"goal": "React Hooks", "points": []})

    kms = list_knowledge_models(store, "u1")
    goals = {km["goal"] for km in kms}
    assert goals == {"Spring AOP", "React Hooks"}


def test_list_knowledge_models_isolation_and_empty():
    """list_knowledge_models 按 user 隔离，空返回 []。"""
    store = InMemoryStore()
    put_knowledge_model(store, "u1", "spring aop", {"goal": "Spring AOP", "points": []})

    assert list_knowledge_models(store, "u2") == []
    assert list_knowledge_models(store, "u1")[0]["goal"] == "Spring AOP"
    assert list_knowledge_models(None, "u1") == []  # store=None 安全降级


# ---- 异步聚合读取（主事件循环内必须用 asearch，防 InvalidStateError）----

def test_alist_knowledge_models_async():
    """alist_knowledge_models 与同步版逻辑一致，走 asearch。"""
    store = InMemoryStore()
    put_knowledge_model(store, "u1", "spring aop", {"goal": "Spring AOP", "points": []})
    put_knowledge_model(store, "u1", "react hooks", {"goal": "React Hooks", "points": []})

    async def go():
        kms = await alist_knowledge_models(store, "u1")
        return {km["goal"] for km in kms}

    assert asyncio.run(go()) == {"Spring AOP", "React Hooks"}


# ---- 观察记录（observation）：AI 观察样本，只追加 ----

def _obs(point_id, observed_state, day, evidence=None):
    """构造带确定 timestamp 的 Observation，保证 key 唯一。"""
    return Observation(
        point_id=point_id,
        observed_state=observed_state,
        confidence=Confidence.HIGH,
        evidence=evidence if evidence is not None else [f"证据-{day}"],
        timestamp=datetime(2026, 9, day, tzinfo=timezone.utc),
    )


def test_record_observation_appends_history():
    """多次记录同一点位观察，历史完整保留（只追加，不覆盖）。"""
    store = InMemoryStore()
    assert record_observation(store, "u1", _obs("p1", CognitiveState.PARTIAL, 1)) is True
    assert record_observation(store, "u1", _obs("p1", CognitiveState.MASTERED, 2)) is True

    history = query_observations(store, "u1", "p1")
    assert len(history) == 2
    assert history[0]["observed_state"] == "partial"
    assert history[1]["observed_state"] == "mastered"


def test_query_observations_sorted_by_timestamp():
    """观察历史按 timestamp 升序返回（可审计）。"""
    store = InMemoryStore()
    record_observation(store, "u1", _obs("p1", CognitiveState.MASTERED, 2))
    record_observation(store, "u1", _obs("p1", CognitiveState.PARTIAL, 1))

    history = query_observations(store, "u1", "p1")
    assert [h["observed_state"] for h in history] == ["partial", "mastered"]


def test_record_observation_skips_unassessed():
    """unassessed 不产生观测（跳过，不落库）。"""
    store = InMemoryStore()
    assert record_observation(store, "u1", _obs("p1", CognitiveState.UNASSESSED, 1)) is False
    assert query_observations(store, "u1", "p1") == []


def test_record_observation_rejects_empty_evidence():
    """evidence 为空（或全空白）时拒绝写入，禁止脑补证据。"""
    store = InMemoryStore()
    assert record_observation(store, "u1", _obs("p1", CognitiveState.PARTIAL, 1, evidence=[])) is False
    assert record_observation(store, "u1", _obs("p1", CognitiveState.PARTIAL, 1, evidence=["  "])) is False
    assert query_observations(store, "u1", "p1") == []


def test_record_observation_store_none_degrades():
    """store 为 None 时安全降级，返回 False 不抛错。"""
    assert record_observation(None, "u1", _obs("p1", CognitiveState.PARTIAL, 1)) is False


def test_query_observations_isolation_and_empty():
    """观察按 user 隔离，store 为 None / 无数据返回空列表。"""
    store = InMemoryStore()
    record_observation(store, "u1", _obs("p1", CognitiveState.PARTIAL, 1))

    assert query_observations(store, "u2", "p1") == []
    assert query_observations(store, "u1", "p2") == []
    assert query_observations(None, "u1", "p1") == []


def test_arecord_observation_async():
    """arecord_observation 异步写入，读回一致。"""
    store = InMemoryStore()
    assert asyncio.run(arecord_observation(store, "u1", _obs("p1", CognitiveState.PARTIAL, 1))) is True
    assert query_observations(store, "u1", "p1")[0]["observed_state"] == "partial"


def test_aquery_observations_async():
    """aquery_observations 与同步版逻辑一致，走 asearch。"""
    store = InMemoryStore()
    record_observation(store, "u1", _obs("p1", CognitiveState.PARTIAL, 1))
    record_observation(store, "u1", _obs("p1", CognitiveState.MASTERED, 2))

    async def go():
        return await aquery_observations(store, "u1", "p1")

    assert [h["observed_state"] for h in asyncio.run(go())] == ["partial", "mastered"]


# ---- 知识结构管理（知识版图能力域 A）----

def _attrs():
    """构造 PointAttributes，供测试复用。"""
    return PointAttributes(
        type=PointType.CONCEPT,
        difficulty=2,
        importance=3,
        bloom_level=BloomLevel.UNDERSTAND,
    )


def _mk_point(point_id, name=None, prerequisites=None, attributes=None):
    """构造带确定字段的 KnowledgePoint。"""
    return KnowledgePoint(
        id=point_id,
        name=name if name is not None else f"点-{point_id}",
        description=f"{point_id} 的描述",
        prerequisites=prerequisites or [],
        attributes=attributes,
    )


def test_insert_point_creates_structure_and_reads_back():
    """insert_point 结构不存在时创建空结构并插入节点。"""
    store = InMemoryStore()
    insert_point(store, "u1", "spring aop", _mk_point("a"))
    km = query_structure(store, "u1", "spring aop")
    assert km is not None
    assert [p.id for p in km.points] == ["a"]


def test_insert_point_idempotent_replace():
    """insert_point 幂等：同 id 节点被替换，不产生重复。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a", name="旧"))
    insert_point(store, "u1", "g", _mk_point("a", name="新"))
    km = query_structure(store, "u1", "g")
    assert len(km.points) == 1
    assert km.points[0].name == "新"


def test_update_point():
    """update_point 更新名称与描述。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    update_point(store, "u1", "g", "a", "新名", "新描述")
    p = query_point(store, "u1", "g", "a")
    assert p.name == "新名"
    assert p.description == "新描述"


def test_update_point_missing_raises():
    """update_point 作用于不存在的点报错。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    with pytest.raises(ValueError):
        update_point(store, "u1", "g", "nonexistent", "x", "y")


def test_delete_point_cascades_edges():
    """delete_point 级联删除其他节点指向它的依赖边。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    insert_point(store, "u1", "g", _mk_point("b", prerequisites=["a"]))
    insert_point(store, "u1", "g", _mk_point("c", prerequisites=["a"]))
    delete_point(store, "u1", "g", "a")

    km = query_structure(store, "u1", "g")
    assert {p.id for p in km.points} == {"b", "c"}
    for p in km.points:
        assert "a" not in p.prerequisites


def test_delete_point_missing_raises():
    """delete_point 作用于不存在的点报错。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    with pytest.raises(ValueError):
        delete_point(store, "u1", "g", "nonexistent")


def test_add_prerequisite_and_query():
    """add_prerequisite 建边后 query_prerequisites 读回。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    insert_point(store, "u1", "g", _mk_point("b"))
    add_prerequisite(store, "u1", "g", "b", "a")  # b 依赖 a
    assert [p.id for p in query_prerequisites(store, "u1", "g", "b")] == ["a"]


def test_add_prerequisite_missing_point_raises():
    """依赖边指向不存在的点报错。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    with pytest.raises(ValueError):
        add_prerequisite(store, "u1", "g", "a", "nonexistent")


def test_add_prerequisite_self_raises():
    """知识点不能依赖自身。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    with pytest.raises(ValueError):
        add_prerequisite(store, "u1", "g", "a", "a")


def test_add_prerequisite_cycle_raises():
    """直接环（a 依赖 b 且 b 依赖 a）拒绝。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    insert_point(store, "u1", "g", _mk_point("b", prerequisites=["a"]))
    with pytest.raises(ValueError):
        add_prerequisite(store, "u1", "g", "a", "b")


def test_add_prerequisite_transitive_cycle_raises():
    """传递环（a → b → c → a）拒绝。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    insert_point(store, "u1", "g", _mk_point("b", prerequisites=["a"]))
    insert_point(store, "u1", "g", _mk_point("c", prerequisites=["b"]))
    with pytest.raises(ValueError):
        add_prerequisite(store, "u1", "g", "a", "c")


def test_remove_prerequisite():
    """remove_prerequisite 解除依赖边（幂等）。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    insert_point(store, "u1", "g", _mk_point("b", prerequisites=["a"]))
    remove_prerequisite(store, "u1", "g", "b", "a")
    assert query_prerequisites(store, "u1", "g", "b") == []


def test_set_attributes_and_query():
    """set_attributes 登记属性后 query_point 读回完整 attributes。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    set_attributes(store, "u1", "g", "a", _attrs())
    p = query_point(store, "u1", "g", "a")
    assert p.attributes.type == PointType.CONCEPT
    assert p.attributes.bloom_level == BloomLevel.UNDERSTAND


def test_query_dependents():
    """query_dependents 返回依赖该点的所有后继。"""
    store = InMemoryStore()
    insert_point(store, "u1", "g", _mk_point("a"))
    insert_point(store, "u1", "g", _mk_point("b", prerequisites=["a"]))
    insert_point(store, "u1", "g", _mk_point("c", prerequisites=["a"]))
    assert {p.id for p in query_dependents(store, "u1", "g", "a")} == {"b", "c"}


def test_query_missing_returns_none_or_empty():
    """结构/点不存在时查询返回 None 或空列表，不抛错。"""
    store = InMemoryStore()
    assert query_structure(store, "u1", "g") is None
    assert query_point(store, "u1", "g", "a") is None
    assert query_prerequisites(store, "u1", "g", "a") == []
    assert query_dependents(store, "u1", "g", "a") == []


def test_structure_ops_store_none_degrades():
    """store 为 None 时写操作安全降级不抛错，查询返回空。"""
    insert_point(None, "u1", "g", _mk_point("a"))
    update_point(None, "u1", "g", "a", "x", "y")
    delete_point(None, "u1", "g", "a")
    add_prerequisite(None, "u1", "g", "a", "b")
    remove_prerequisite(None, "u1", "g", "a", "b")
    set_attributes(None, "u1", "g", "a", _attrs())
    assert query_structure(None, "u1", "g") is None


def test_structure_requires_existing_structure():
    """写操作（insert 除外）在结构不存在时报错。"""
    store = InMemoryStore()
    with pytest.raises(ValueError):
        update_point(store, "u1", "g", "a", "x", "y")


# ---- 权威熟练度查询（query_proficiency）----

def test_query_proficiency_empty_unassessed():
    """无观察 → unassessed，且 point_id 正确补回。"""
    store = InMemoryStore()
    prof = query_proficiency(store, "u1", "p1")
    assert prof is not None
    assert prof.point_id == "p1"
    assert prof.mapped_state == CognitiveState.UNASSESSED
    assert prof.observation_count == 0


def test_query_proficiency_with_observations():
    """有观察 → 返回 BKT 融合后的权威状态。"""
    store = InMemoryStore()
    record_observation(store, "u1", _obs("p1", CognitiveState.MASTERED, 1))
    prof = query_proficiency(store, "u1", "p1")
    assert prof is not None
    assert prof.point_id == "p1"
    assert prof.observation_count == 1


def test_query_proficiency_store_none_degrades():
    """store 为 None → 返回 None（安全降级）。"""
    assert query_proficiency(None, "u1", "p1") is None


def test_query_proficiency_missing_point_unassessed():
    """不存在的 point → 返回 unassessed（不抛错，与 unknown 区分）。"""
    store = InMemoryStore()
    record_observation(store, "u1", _obs("p1", CognitiveState.MASTERED, 1))
    prof = query_proficiency(store, "u1", "nonexistent")
    assert prof is not None
    assert prof.point_id == "nonexistent"
    assert prof.mapped_state == CognitiveState.UNASSESSED


def test_query_proficiency_user_isolation():
    """不同 user 隔离：u1 的观察对 u2 不可见。"""
    store = InMemoryStore()
    record_observation(store, "u1", _obs("p1", CognitiveState.MASTERED, 1))
    prof = query_proficiency(store, "u2", "p1")
    assert prof.mapped_state == CognitiveState.UNASSESSED
    assert prof.observation_count == 0


def test_aquery_proficiency_async():
    """aquery_proficiency 与同步版逻辑一致。"""
    store = InMemoryStore()
    record_observation(store, "u1", _obs("p1", CognitiveState.MASTERED, 1))

    async def go():
        return await aquery_proficiency(store, "u1", "p1")

    prof = asyncio.run(go())
    assert prof.point_id == "p1"
    assert prof.observation_count == 1


# ---- 权威熟练度聚合（alist_authoritative_proficiencies）----

def _km_with_point(pid, attributes=None):
    """构造含单个 point 的知识模型 dict（attributes 为 PointAttributes 或 None）。"""
    point = {"id": pid, "name": f"点-{pid}", "description": f"{pid} 描述"}
    if attributes is not None:
        point["attributes"] = attributes.model_dump(mode="json")
    return {"goal": "g", "points": [point]}


def _attrs_bloom(bloom):
    """按认知层级构造 PointAttributes。"""
    return PointAttributes(
        type=PointType.CONCEPT,
        difficulty=2,
        importance=3,
        bloom_level=bloom,
    )


def test_alist_authoritative_proficiencies_empty():
    """无数据 → 返回 {}。"""
    store = InMemoryStore()

    async def go():
        return await alist_authoritative_proficiencies(store, "u1")

    assert asyncio.run(go()) == {}


def test_alist_authoritative_proficiencies_mastered_understand():
    """单次 mastered + understand（depth=1）→ mastered。"""
    store = InMemoryStore()
    put_knowledge_model(
        store, "u1", "g",
        _km_with_point("p1", _attrs_bloom(BloomLevel.UNDERSTAND)),
    )
    record_observation(store, "u1", _obs("p1", CognitiveState.MASTERED, 1))

    async def go():
        return await alist_authoritative_proficiencies(store, "u1")

    assert asyncio.run(go()) == {"p1": "mastered"}


def test_alist_authoritative_proficiencies_apply_requires_two():
    """单次 mastered + apply（depth=2）→ 观察数不足，不定 mastered（partial）。"""
    store = InMemoryStore()
    put_knowledge_model(
        store, "u1", "g",
        _km_with_point("p1", _attrs_bloom(BloomLevel.APPLY)),
    )
    record_observation(store, "u1", _obs("p1", CognitiveState.MASTERED, 1))

    async def go():
        return await alist_authoritative_proficiencies(store, "u1")

    assert asyncio.run(go()) == {"p1": "partial"}


def test_alist_authoritative_proficiencies_no_attributes_fallback():
    """point 无 attributes → 保守 depth=2，单次 mastered 不定 mastered。"""
    store = InMemoryStore()
    put_knowledge_model(store, "u1", "g", _km_with_point("p1", None))
    record_observation(store, "u1", _obs("p1", CognitiveState.MASTERED, 1))

    async def go():
        return await alist_authoritative_proficiencies(store, "u1")

    assert asyncio.run(go()) == {"p1": "partial"}


def test_alist_authoritative_proficiencies_misconception():
    """最近一次负向观察为 misconception → misconception。"""
    store = InMemoryStore()
    put_knowledge_model(
        store, "u1", "g",
        _km_with_point("p1", _attrs_bloom(BloomLevel.UNDERSTAND)),
    )
    record_observation(store, "u1", _obs("p1", CognitiveState.MASTERED, 1))
    record_observation(store, "u1", _obs("p1", CognitiveState.MISCONCEPTION, 2))

    async def go():
        return await alist_authoritative_proficiencies(store, "u1")

    assert asyncio.run(go()) == {"p1": "misconception"}


def test_alist_authoritative_proficiencies_store_none():
    """store 为 None → 返回 {}（安全降级）。"""

    async def go():
        return await alist_authoritative_proficiencies(None, "u1")

    assert asyncio.run(go()) == {}
