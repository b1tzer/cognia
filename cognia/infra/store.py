"""Cognia 生产存储工厂（Postgres 连接池 / Checkpointer / Store）。

把进程级共享的连接池、checkpointer、store 工厂从 memory.py 抽离到基础设施层：
这是「部署期 infra」，与 memory.py 的领域存储逻辑（observation / profile /
knowledge_model 等 store 操作）不同层。

单例 + 共享池：整个进程只建一个 AsyncConnectionPool，Checkpointer 与 Store 复用
同一个池；否则连接数会随会话数线性增长、耗尽 Supabase session pool。
"""

import os

from langgraph.store.base import BaseStore

# 模块级单例缓存：生产环境整个进程只建一个共享 AsyncConnectionPool，并让
# Checkpointer 与 Store 复用同一个池。否则每次请求都新建 ConnectionPool 会导致
# 连接数随会话数线性增长、耗尽 Supabase session pool
# （pool_size=15，报 EMAXCONNSESSION）；同时多个 AsyncConnectionPool 并存会互相
# 竞争，偶发卡死在 pool.open()/setup()。
_pool_cache = None
_checkpointer_cache = None
_store_cache = None


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


async def get_pool():
    """公开的进程级 AsyncConnectionPool 访问（线程元数据层复用同一连接池）。

    与 checkpointer / store 共享同一个池（见 _get_pool 说明），避免多池竞争。
    线程元数据（cognia_threads 表）也走这个池，保证与会话 checkpoint 同库同池。
    """
    return await _get_pool()


async def get_checkpointer():
    """生产 Postgres Checkpointer（按 thread_id 恢复会话）。

    **必须返回 AsyncPostgresSaver**：服务端用 `graph.astream()` 异步执行，
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
    """生产 Postgres Store（proficiency / profile / knowledge_model 多命名空间）。

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
