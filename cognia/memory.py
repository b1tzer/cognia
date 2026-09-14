"""Cognia 记忆层（短期 Checkpointer + 长期 Store）。

职责边界（plan §3.3、clarifications Q6）：
- Checkpointer：短期会话状态，按 `thread_id` 恢复（一次会话）。
- Store：长期认知状态，按 `user_id` 隔离，跨会话持久化。
  - `("proficiency", user_id)`：动态熟练度（增量 Delta，严禁全量重写，宪法 §5）。
  - `("profile", user_id)`：稳定偏好（沟通风格 / 语言）。

匿名 `user_id` 通过 runtime context（`config["configurable"]["user_id"]`）注入，
绝不塞进 State（宪法 §5）。前端生成 UUID 并持久化，后端仅作隔离映射。

本模块的「逻辑函数」（append / get 等）接受 `store` 参数、不自行建连接，
便于测试用 `InMemoryStore` 注入；「生产工厂」（get_checkpointer / get_store）
读 `LANGGRAPH_DATABASE_URL` 建立 Postgres 连接，仅在部署时使用。
"""

import os

from langgraph.store.base import BaseStore

from cognia.schemas import ProficiencyEntry

# namespace 第一段（Store 的 namespace 是 tuple：类别 + user_id）
PROFICIENCY_NS = "proficiency"
PROFILE_NS = "profile"

# 模块级单例缓存：生产环境整个进程只建一个共享 AsyncConnectionPool，并让
# Checkpointer 与 Store 复用同一个池。否则 Chainlit 每次 on_chat_start 都新建
# ConnectionPool 会导致连接数随会话数线性增长、耗尽 Supabase session pool
# （pool_size=15，报 EMAXCONNSESSION）；同时多个 AsyncConnectionPool 并存会互相
# 竞争，偶发卡死在 pool.open()/setup()。
_pool_cache = None
_checkpointer_cache = None
_store_cache = None


# ---- 熟练度（proficiency）：增量 Delta 读写 ----

def append_proficiency_delta(store: BaseStore, user_id: str, entry: ProficiencyEntry) -> None:
    """熟练度增量 Delta 追加（append，不覆盖，宪法 §5）。

    每个 Delta 用「point_id + timestamp」做唯一 key，历史不可变、可审计。
    namespace = ("proficiency", user_id)，天然按 user 隔离。
    """
    key = f"{entry.point_id}:{entry.timestamp.isoformat()}"
    store.put((PROFICIENCY_NS, user_id), key, entry.model_dump(mode="json"))


async def aappend_proficiency_delta(store, user_id: str, entry: ProficiencyEntry) -> None:
    """append_proficiency_delta 的异步版本（供 AsyncPostgresStore 在事件循环内调用）。

    同步 `store.put` 在 AsyncPostgresStore 上会因 @_check_loop 装饰器在主事件循环
    线程里抛 InvalidStateError，因此 `_persist_deltas`（Chainlit on_message，主线程）
    必须改用 `await store.aput`。
    """
    key = f"{entry.point_id}:{entry.timestamp.isoformat()}"
    await store.aput((PROFICIENCY_NS, user_id), key, entry.model_dump(mode="json"))


def get_proficiency_history(store: BaseStore, user_id: str, point_id: str) -> list[dict]:
    """读某 user 某知识点的完整 Delta 历史（按 timestamp 升序）。"""
    items = store.search((PROFICIENCY_NS, user_id))
    deltas = [item.value for item in items if item.value.get("point_id") == point_id]
    deltas.sort(key=lambda d: d["timestamp"])
    return deltas


def get_current_proficiency(store: BaseStore, user_id: str, point_id: str) -> str | None:
    """读某 user 某知识点的当前熟练度状态（最新 Delta 的 to_state）。

    None 表示从未评估（unassessed）。
    """
    history = get_proficiency_history(store, user_id, point_id)
    return history[-1]["to_state"] if history else None


# ---- 画像（profile）：基础偏好读写 ----

def put_profile(store: BaseStore, user_id: str, key: str, value: dict) -> None:
    """写入画像字段（如沟通风格、语言偏好）。"""
    store.put((PROFILE_NS, user_id), key, value)


def get_profile(store: BaseStore, user_id: str, key: str) -> dict | None:
    """读取画像字段，不存在返回 None。"""
    item = store.get((PROFILE_NS, user_id), key)
    return item.value if item else None


# ---- runtime context：user_id 注入（不塞 State）----

def get_user_id(config: dict) -> str | None:
    """从 runtime context 提取匿名 user_id。

    config 形如 `{"configurable": {"thread_id": ..., "user_id": ...}}`。
    未提供 user_id 时返回 None。
    """
    return config.get("configurable", {}).get("user_id")


# ---- 生产工厂（读 LANGGRAPH_DATABASE_URL；测试用 InMemoryStore 注入）----

def _require_database_url() -> str:
    conn_string = os.getenv("LANGGRAPH_DATABASE_URL")
    if not conn_string:
        raise RuntimeError("LANGGRAPH_DATABASE_URL 环境变量未设置，无法初始化 Postgres 记忆层")
    return conn_string


async def _get_pool():
    """进程级共享 AsyncConnectionPool（Checkpointer 与 Store 复用同一个池）。

    关键：多个 AsyncConnectionPool 并存于同一事件循环会互相竞争、偶发卡死
    （实测第二个池 open()/setup() 挂起），因此必须共享单池。AsyncConnectionPool
    本身协程安全，多个消费者（Saver / Store）可安全共用。
    """
    global _pool_cache
    if _pool_cache is not None:
        return _pool_cache

    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    # autocommit=True 是关键：langgraph 的 put() 从不显式 commit，写操作依赖连接的
    # autocommit 立即提交。缺了它，连接归还池时事务被回滚，checkpoint / Delta 丢失。
    # prepare_threshold=0 / row_factory=dict_row 对齐官方 from_conn_string 的默认。
    # min_size=1 / max_size=8：Supabase session pool 上限 15，共享单池留足余量。
    pool = AsyncConnectionPool(
        _require_database_url(),
        min_size=1,
        max_size=8,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open(wait=True)
    _pool_cache = pool
    return pool


async def get_checkpointer():
    """生产 Postgres Checkpointer（按 thread_id 恢复会话）。

    **必须返回 AsyncPostgresSaver**：`app.py` 用 `graph.astream()` 异步执行，
    LangGraph 的 AsyncPregelLoop 会调用 `checkpointer.aget_tuple()`；而同步的
    `PostgresSaver` 只实现了同步 `get_tuple`、未实现 `aget_tuple`（基类直接抛
    `NotImplementedError`，且 str 为空），这正是「每次都报错、从未正常对话」的根因。

    单例 + 共享池：与 Store 复用同一个 AsyncConnectionPool（见 _get_pool）。
    """
    global _checkpointer_cache
    if _checkpointer_cache is not None:
        return _checkpointer_cache

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    pool = await _get_pool()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    _checkpointer_cache = checkpointer
    return checkpointer


async def get_store() -> BaseStore:
    """生产 Postgres Store（proficiency / profile 双命名空间）。

    依赖 `psycopg_pool`。返回已 setup 的 AsyncPostgresStore（与 Checkpointer 共享
    同一个 AsyncConnectionPool）。

    必须异步化：同步 PostgresStore 与异步 AsyncPostgresSaver 混在同一事件循环里
    会不稳定；统一用 AsyncConnectionPool 后，graph 同步节点在 executor 线程里调用
    同步 store.search（AsyncPostgresStore 用 run_coroutine_threadsafe 桥接），天然
    线程安全。

    单例 + 共享池：与 Checkpointer 复用同一个池（见 _get_pool）。
    """
    global _store_cache
    if _store_cache is not None:
        return _store_cache

    from langgraph.store.postgres.aio import AsyncPostgresStore

    pool = await _get_pool()
    store = AsyncPostgresStore(pool)
    await store.setup()
    _store_cache = store
    return store
