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
读 `DATABASE_URL` 建立 Postgres 连接，仅在部署时使用。
"""

import os

from langgraph.store.base import BaseStore

from cognia.schemas import ProficiencyEntry

# namespace 第一段（Store 的 namespace 是 tuple：类别 + user_id）
PROFICIENCY_NS = "proficiency"
PROFILE_NS = "profile"


# ---- 熟练度（proficiency）：增量 Delta 读写 ----

def append_proficiency_delta(store: BaseStore, user_id: str, entry: ProficiencyEntry) -> None:
    """熟练度增量 Delta 追加（append，不覆盖，宪法 §5）。

    每个 Delta 用「point_id + timestamp」做唯一 key，历史不可变、可审计。
    namespace = ("proficiency", user_id)，天然按 user 隔离。
    """
    key = f"{entry.point_id}:{entry.timestamp.isoformat()}"
    store.put((PROFICIENCY_NS, user_id), key, entry.model_dump(mode="json"))


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


# ---- 生产工厂（读 DATABASE_URL；测试用 InMemoryStore 注入）----

def _require_database_url() -> str:
    conn_string = os.getenv("DATABASE_URL")
    if not conn_string:
        raise RuntimeError("DATABASE_URL 环境变量未设置，无法初始化 Postgres 记忆层")
    return conn_string


def get_checkpointer():
    """生产 Postgres Checkpointer（按 thread_id 恢复会话）。

    依赖 `psycopg_pool`（随 langgraph-checkpoint-postgres 安装）。
    返回已 setup 的 PostgresSaver（内部持有连接池）。
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    # autocommit=True 是关键：langgraph 的 put() 从不显式 commit，写操作依赖连接的
    # autocommit 立即提交。缺了它，连接归还池时事务被回滚，checkpoint 静默丢失。
    # prepare_threshold=0 / row_factory=dict_row 对齐官方 from_conn_string 的默认。
    pool = ConnectionPool(
        _require_database_url(),
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    return checkpointer


def get_store() -> BaseStore:
    """生产 Postgres Store（proficiency / profile 双命名空间）。

    依赖 `psycopg_pool`。返回已 setup 的 PostgresStore（内部持有连接池）。
    """
    from langgraph.store.postgres import PostgresStore
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    # autocommit=True 是关键：langgraph 的 put() 从不显式 commit，写操作依赖连接的
    # autocommit 立即提交。缺了它，连接归还池时事务被回滚，熟练度 Delta 静默丢失。
    # prepare_threshold=0 / row_factory=dict_row 对齐官方 from_conn_string 的默认。
    pool = ConnectionPool(
        _require_database_url(),
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    store = PostgresStore(pool)
    store.setup()
    return store
