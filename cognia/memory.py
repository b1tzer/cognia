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

from cognia.schemas import (
    CognitiveState,
    KnowledgeModel,
    KnowledgePoint,
    Observation,
    PointAttributes,
    Proficiency,
)

from cognia.proficiency_engine import compute_proficiency
from cognia import concept_merge, embedding

# namespace 第一段（Store 的 namespace 是 tuple：类别 + user_id）
PROFILE_NS = "profile"
KNOWLEDGE_MODEL_NS = "knowledge_model"
OBSERVATION_NS = "observation"
CONCEPT_NS = "concept"

# 模块级单例缓存：生产环境整个进程只建一个共享 AsyncConnectionPool，并让
# Checkpointer 与 Store 复用同一个池。否则每次请求都新建 ConnectionPool 会导致
# 连接数随会话数线性增长、耗尽 Supabase session pool
# （pool_size=15，报 EMAXCONNSESSION）；同时多个 AsyncConnectionPool 并存会互相
# 竞争，偶发卡死在 pool.open()/setup()。
_pool_cache = None
_checkpointer_cache = None
_store_cache = None


# ---- 观察记录（observation）：AI 观察样本，只追加 ----

def _has_evidence(observation: Observation) -> bool:
    """evidence 过滤空白后是否非空（严禁脑补证据，spec §6）。"""
    return any(str(e).strip() for e in (observation.evidence or []))


def record_observation(store, user_id: str, observation: Observation) -> bool:
    """追加一条 AI 观察样本（只追加，不直接改权威状态）。

    - store / user_id 缺失 → 安全降级，返回 False（不落库）。
    - observed_state == unassessed → 跳过（未评估不产生观测）。
    - evidence 为空 → 拒绝写入（严禁脑补证据，spec §6）。

    返回 True 表示已记录，False 表示跳过。key = point_id + timestamp，
    历史不可变、可审计。namespace = ("observation", user_id)。
    """
    if store is None or not user_id:
        return False
    if observation.observed_state == CognitiveState.UNASSESSED:
        return False
    if not _has_evidence(observation):
        return False
    key = f"{observation.point_id}:{observation.timestamp.isoformat()}"
    store.put((OBSERVATION_NS, user_id), key, observation.model_dump(mode="json"))
    return True


async def arecord_observation(store, user_id: str, observation: Observation) -> bool:
    """record_observation 的异步版本（供主事件循环内调用）。

    AsyncPostgresStore 在主事件循环线程里必须用 `await store.aput`。
    """
    if store is None or not user_id:
        return False
    if observation.observed_state == CognitiveState.UNASSESSED:
        return False
    if not _has_evidence(observation):
        return False
    key = f"{observation.point_id}:{observation.timestamp.isoformat()}"
    await store.aput((OBSERVATION_NS, user_id), key, observation.model_dump(mode="json"))
    return True


def query_observations(store, user_id: str, point_id: str) -> list[dict]:
    """读某 user 某知识点的完整观察历史（按 timestamp 升序，可审计）。"""
    if store is None or not user_id or not point_id:
        return []
    items = store.search((OBSERVATION_NS, user_id))
    obs = [item.value for item in items if item.value.get("point_id") == point_id]
    obs.sort(key=lambda d: d.get("timestamp") or "")
    return obs


async def aquery_observations(store, user_id: str, point_id: str) -> list[dict]:
    """query_observations 的异步版本（主事件循环内用 asearch）。"""
    if store is None or not user_id or not point_id:
        return []
    items = await store.asearch((OBSERVATION_NS, user_id))
    obs = [item.value for item in items if item.value.get("point_id") == point_id]
    obs.sort(key=lambda d: d.get("timestamp") or "")
    return obs


# ---- 权威熟练度（proficiency）查询：观察历史 → BKT 融合 → Proficiency ----

def query_proficiency(store, user_id: str, point_id: str) -> Proficiency | None:
    """读某 user 某知识点的权威熟练度（能力域 D：单点查询）。

    内部：query_observations → compute_proficiency → Proficiency。
    这是系统计算产物，AI 只能查询、无权直接改写。

    - store / user_id / point_id 缺失 → 返回 None（安全降级）。
    - 无观察 → 返回 point_id 正确、mapped_state=unassessed 的 Proficiency。
    - 单点查询无 goal 上下文，attributes 传 None（难度回退默认 P(T)、
      bloom_level 保守取 depth=2）；整图查询（#109）在有 goal 上下文时另行传属性。
    """
    if store is None or not user_id or not point_id:
        return None
    observations = query_observations(store, user_id, point_id)
    prof = compute_proficiency(observations, None)
    if prof.point_id == "":
        # 无观察时 compute_proficiency 返回空 point_id，这里补回真实 point_id
        return prof.model_copy(update={"point_id": point_id})
    return prof


async def aquery_proficiency(store, user_id: str, point_id: str) -> Proficiency | None:
    """query_proficiency 的异步版本（主事件循环内用 asearch）。"""
    if store is None or not user_id or not point_id:
        return None
    observations = await aquery_observations(store, user_id, point_id)
    prof = compute_proficiency(observations, None)
    if prof.point_id == "":
        return prof.model_copy(update={"point_id": point_id})
    return prof


# ---- 画像（profile）：基础偏好读写 ----

def put_profile(store: BaseStore, user_id: str, key: str, value: dict) -> None:
    """写入画像字段（如沟通风格、语言偏好）。"""
    store.put((PROFILE_NS, user_id), key, value)


def get_profile(store: BaseStore, user_id: str, key: str) -> dict | None:
    """读取画像字段，不存在返回 None。"""
    item = store.get((PROFILE_NS, user_id), key)
    return item.value if item else None


def get_profile_dict(store: BaseStore, user_id: str) -> dict:
    """读取某 user 的全部画像字段，合并为扁平 dict。

    用于对话 Agent 注入 system prompt（沟通风格 / 语言偏好 / 学习偏好等）。
    各字段值约定为 `{"value": ...}` 结构，这里取 `value` 便于直接拼接。
    """
    if store is None or not user_id:
        return {}
    items = store.search((PROFILE_NS, user_id))
    profile: dict = {}
    for item in items:
        value = item.value
        if isinstance(value, dict):
            profile[item.key] = value.get("value", value)
        else:
            profile[item.key] = value
    return profile


async def aput_profile(store, user_id: str, key: str, value: dict) -> None:
    """put_profile 的异步版本（供主事件循环内调用）。

    AsyncPostgresStore 在主事件循环线程里必须用 `await store.aput`，
    同步 `store.put` 会抛 InvalidStateError。
    """
    await store.aput((PROFILE_NS, user_id), key, value)


# ---- 知识模型（knowledge_model）：load-or-build 冻结持久化 ----

def normalize_goal(goal: str) -> str:
    """归一化学习目标，作为知识模型持久化 key（Task ⑨）。

    MVP 只做「去首尾空白 + 折叠内部空白 + 统一小写」，不做语义同义映射
    （同义映射需向量 / 词典，误合并风险高，留待后续）。归一化后等价目标
    （如「Spring AOP」「spring  aop」「SPRING AOP」）映射到同一 key，
    保证知识点 identity（point_id）跨会话稳定。
    """
    import re
    return re.sub(r"\s+", " ", goal.strip().lower())


def put_knowledge_model(store: BaseStore, user_id: str, goal_key: str, km: dict) -> None:
    """冻结持久化某 user 的知识模型（首次构建后不再重生成）。

    namespace = ("knowledge_model", user_id)，key = 归一化 goal，天然按 user 隔离。
    知识模型是「领域模型」，与熟练度（proficiency）/ 偏好（profile）分 namespace。
    只有 graph 节点（executor 线程）调用，故用同步 store.put 即可（AsyncPostgresStore
    会在 executor 线程内桥接，见 get_store 说明）。
    """
    store.put((KNOWLEDGE_MODEL_NS, user_id), goal_key, km)


def get_knowledge_model(store: BaseStore, user_id: str, goal_key: str) -> dict | None:
    """读取已冻结的知识模型，不存在返回 None（触发重新构建）。"""
    item = store.get((KNOWLEDGE_MODEL_NS, user_id), goal_key)
    return item.value if item else None


# ---- 知识结构管理（知识版图能力域 A：节点 CRUD + 依赖边 + 属性）----
#
# 复用 knowledge_model namespace 作为「知识结构」的持久化载体：同一份数据
# （KnowledgeModel = goal + points），既支持 build_learning_goal 的批量生成，
# 也支持这里的细粒度增删改。point_id 在单个 goal 内唯一，故以 goal_key 定位。

def _load_structure(store, user_id: str, goal_key: str) -> KnowledgeModel | None:
    """读某 user 某 goal 的知识结构（KnowledgeModel），不存在返回 None。"""
    km_dict = get_knowledge_model(store, user_id, goal_key)
    if km_dict is None:
        return None
    return KnowledgeModel.model_validate(km_dict)


def _save_structure(store, user_id: str, goal_key: str, km: KnowledgeModel) -> None:
    """写回某 user 某 goal 的知识结构（整体覆盖该 goal 的 value）。"""
    put_knowledge_model(store, user_id, goal_key, km.model_dump(mode="json"))


def _require_structure(store, user_id: str, goal_key: str) -> KnowledgeModel:
    """读知识结构，结构不存在则报错（写/查询操作的前置守卫）。"""
    km = _load_structure(store, user_id, goal_key)
    if km is None:
        raise ValueError(f"知识结构不存在：goal_key={goal_key}")
    return km


def _find_point(km: KnowledgeModel, point_id: str) -> KnowledgePoint | None:
    """在知识结构中按 point_id 查找知识点，不存在返回 None。"""
    return next((p for p in km.points if p.id == point_id), None)


def _would_create_cycle(point_id: str, prerequisite_id: str, points: list[KnowledgePoint]) -> bool:
    """加边「point_id 依赖 prerequisite_id」是否形成环。

    判断 prerequisite_id 是否已（直接或间接）依赖 point_id：沿 prerequisites
    反向 DFS，若能到达 point_id 则加边后形成环。
    """
    by_id = {p.id: p for p in points}
    visited: set[str] = set()
    stack = [prerequisite_id]
    while stack:
        cur = stack.pop()
        if cur == point_id:
            return True
        if cur in visited:
            continue
        visited.add(cur)
        p = by_id.get(cur)
        if p:
            stack.extend(p.prerequisites)
    return False


def insert_point(store, user_id: str, goal_key: str, point: KnowledgePoint) -> None:
    """新增知识点节点（幂等：point_id 已存在则替换该节点）。

    store / user_id 缺失 → 安全降级（不落库，不抛错）。
    结构不存在 → 以 goal_key 为 goal 创建空结构后插入。
    """
    if store is None or not user_id:
        return
    km = _load_structure(store, user_id, goal_key)
    if km is None:
        km = KnowledgeModel(goal=goal_key, points=[])
    # 幂等替换：先剔除同 id 旧节点，再追加新节点
    km.points = [p for p in km.points if p.id != point.id] + [point]
    _save_structure(store, user_id, goal_key, km)


def update_point(store, user_id: str, goal_key: str, point_id: str, name: str, description: str) -> None:
    """更新知识点名称与描述（点不存在报错）。"""
    if store is None or not user_id:
        return
    km = _require_structure(store, user_id, goal_key)
    point = _find_point(km, point_id)
    if point is None:
        raise ValueError(f"知识点不存在：point_id={point_id}")
    point.name = name
    point.description = description
    _save_structure(store, user_id, goal_key, km)


def delete_point(store, user_id: str, goal_key: str, point_id: str) -> None:
    """删除知识点节点，并级联删除其他节点指向它的依赖边。"""
    if store is None or not user_id:
        return
    km = _require_structure(store, user_id, goal_key)
    if _find_point(km, point_id) is None:
        raise ValueError(f"知识点不存在：point_id={point_id}")
    # 删除节点本身
    km.points = [p for p in km.points if p.id != point_id]
    # 级联删除其他节点对它的依赖引用
    for p in km.points:
        p.prerequisites = [pid for pid in p.prerequisites if pid != point_id]
    _save_structure(store, user_id, goal_key, km)


def add_prerequisite(store, user_id: str, goal_key: str, point_id: str, prerequisite_id: str) -> None:
    """建立「point_id 依赖 prerequisite_id」的依赖边。

    - 依赖边指向不存在的点 → 报错。
    - 引入环 → 拒绝（保持 DAG 合法）。
    - 自依赖（point_id == prerequisite_id）→ 拒绝。
    """
    if store is None or not user_id:
        return
    km = _require_structure(store, user_id, goal_key)
    if _find_point(km, point_id) is None:
        raise ValueError(f"知识点不存在：point_id={point_id}")
    if _find_point(km, prerequisite_id) is None:
        raise ValueError(f"依赖知识点不存在：prerequisite_id={prerequisite_id}")
    if point_id == prerequisite_id:
        raise ValueError("知识点不能依赖自身")
    if _would_create_cycle(point_id, prerequisite_id, km.points):
        raise ValueError(f"加边会形成环：{point_id} → {prerequisite_id}")
    point = _find_point(km, point_id)
    if prerequisite_id not in point.prerequisites:
        point.prerequisites = [*point.prerequisites, prerequisite_id]
        _save_structure(store, user_id, goal_key, km)


def remove_prerequisite(store, user_id: str, goal_key: str, point_id: str, prerequisite_id: str) -> None:
    """解除「point_id 依赖 prerequisite_id」的依赖边（幂等：边不存在不报错）。"""
    if store is None or not user_id:
        return
    km = _require_structure(store, user_id, goal_key)
    point = _find_point(km, point_id)
    if point is None:
        raise ValueError(f"知识点不存在：point_id={point_id}")
    point.prerequisites = [pid for pid in point.prerequisites if pid != prerequisite_id]
    _save_structure(store, user_id, goal_key, km)


def set_attributes(store, user_id: str, goal_key: str, point_id: str, attributes: PointAttributes) -> None:
    """登记知识点本体属性（点不存在报错）。"""
    if store is None or not user_id:
        return
    km = _require_structure(store, user_id, goal_key)
    point = _find_point(km, point_id)
    if point is None:
        raise ValueError(f"知识点不存在：point_id={point_id}")
    point.attributes = attributes
    _save_structure(store, user_id, goal_key, km)


def query_structure(store, user_id: str, goal_key: str) -> KnowledgeModel | None:
    """读某 user 某 goal 的完整知识结构（不存在返回 None）。"""
    if store is None or not user_id:
        return None
    return _load_structure(store, user_id, goal_key)


def query_point(store, user_id: str, goal_key: str, point_id: str) -> KnowledgePoint | None:
    """读某知识点（不存在返回 None）。"""
    km = query_structure(store, user_id, goal_key)
    if km is None:
        return None
    return _find_point(km, point_id)


def query_prerequisites(store, user_id: str, goal_key: str, point_id: str) -> list[KnowledgePoint]:
    """读某知识点的全部前置依赖点（按 prerequisites 顺序）。"""
    km = query_structure(store, user_id, goal_key)
    if km is None:
        return []
    point = _find_point(km, point_id)
    if point is None:
        return []
    by_id = {p.id: p for p in km.points}
    return [by_id[pid] for pid in point.prerequisites if pid in by_id]


def query_dependents(store, user_id: str, goal_key: str, point_id: str) -> list[KnowledgePoint]:
    """读某知识点的全部后继（依赖它的点）。"""
    km = query_structure(store, user_id, goal_key)
    if km is None:
        return []
    return [p for p in km.points if point_id in p.prerequisites]


# ---- 全局概念注册表（concept）：跨对话概念身份归一 ----

def put_concept(store, user_id: str, global_id: str, name: str, embedding_vec: list[float] | None) -> None:
    """写 / 更新一个全局概念（key = global_id，value 含 name + embedding）。

    store / user_id 缺失 → 安全降级（不落库）。embedding 为 None 时存空列表。
    namespace = ("concept", user_id)，天然按 user 隔离。
    """
    if store is None or not user_id:
        return
    store.put((CONCEPT_NS, user_id), global_id, {
        "id": global_id,
        "name": name,
        "embedding": embedding_vec or [],
    })


def list_concepts(store, user_id: str) -> list[dict]:
    """读某 user 的全部全局概念（每个元素 {id, name, embedding}）。"""
    if store is None or not user_id:
        return []
    items = store.search((CONCEPT_NS, user_id))
    return [item.value for item in items if item.value is not None]


def merge_knowledge_model_concepts(store, user_id: str, km: KnowledgeModel) -> KnowledgeModel:
    """对知识模型的全部点做语义合并，重写 point_id 为全局稳定 id（原地 + 返回）。

    - 读 user 已有全局概念 → 每个点按 name 语义合并（同义不同名合并到同一 id）。
    - 新概念计算 embedding 落库（concept namespace）。
    - 同步重写 point.prerequisites 里的裸 id 引用为全局 id。
    - store / user_id 缺失 → 保持原样返回（不合并，安全降级）。

    这是「以 user 为单位构建知识星图」的落点：同一概念跨对话复用同一 point_id，
    熟练度 / 观察历史按全局 id 自然聚合。
    """
    if store is None or not user_id:
        return km

    existing = list_concepts(store, user_id)
    id_map: dict[str, str] = {}

    for p in km.points:
        global_id, is_new, canonical = concept_merge.merge_concept(p.name, existing)
        id_map[p.id] = global_id
        if is_new:
            vec = embedding.embed_text(canonical)
            put_concept(store, user_id, global_id, canonical, vec)
            existing.append({"id": global_id, "name": canonical, "embedding": vec})
        p.id = global_id

    # 重写前置依赖引用：旧裸 id → 全局 id（跨 goal 的历史点 id 不在本批，保持不变）
    for p in km.points:
        p.prerequisites = [id_map.get(pid, pid) for pid in p.prerequisites]

    return km


# ---- 聚合读取（供个人页 / 知识版图拉取，纯函数无副作用）----

def list_knowledge_models(store: BaseStore, user_id: str) -> list[dict]:
    """读某 user 的全部知识模型（按 goal 冻结的 value 列表）。

    store 为 None / user_id 缺失时返回空列表（安全降级，不抛错）。
    每个元素是知识模型 dict（含 goal / points）。
    """
    if store is None or not user_id:
        return []
    items = store.search((KNOWLEDGE_MODEL_NS, user_id))
    return [item.value for item in items if item.value is not None]


async def alist_knowledge_models(store, user_id: str) -> list[dict]:
    """list_knowledge_models 的异步版本（供主事件循环内的 async endpoint 调用）。

    AsyncPostgresStore 在主事件循环线程里禁止同步 store.search（抛
    asyncio.InvalidStateError），必须 `await store.asearch(...)`。逻辑与同步版一致。
    """
    if store is None or not user_id:
        return []
    items = await store.asearch((KNOWLEDGE_MODEL_NS, user_id))
    return [item.value for item in items if item.value is not None]


async def alist_authoritative_proficiencies(store, user_id: str) -> dict[str, str]:
    """读某 user 全部知识点的权威熟练度（观察历史 → BKT 融合 → mapped_state）。

    本函数读 observation 样本并经 BKT 算法融合出权威状态——知识版图子系统的
    核心语义「AI 只提交观察值，系统算法定级」。

    返回 `{point_id: mapped_state}`；从未评估过的 point 不在结果里（由调用方补
    unassessed）。遍历 knowledge_model 的每个 point，用其 attributes（难度 /
    认知层级）驱动 BKT 参数与验证深度；attributes 缺失或非法时回退默认值。
    """
    if store is None or not user_id:
        return {}
    kms = await alist_knowledge_models(store, user_id)
    result: dict[str, str] = {}
    for km in kms:
        for p in (km.get("points") or []):
            pid = p.get("id")
            if not pid:
                continue
            attributes = None
            attrs = p.get("attributes")
            if isinstance(attrs, dict):
                try:
                    attributes = PointAttributes.model_validate(attrs)
                except Exception:
                    attributes = None
            obs = await aquery_observations(store, user_id, pid)
            prof = compute_proficiency(obs, attributes)
            result[pid] = prof.mapped_state.value
    return result


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
