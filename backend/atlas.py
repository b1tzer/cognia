"""全局概念库（Cognitive Atlas）归一化匹配层。

职责：把每次 LLM 拆解出的概念，通过「规范化名」匹配到全局概念库的稳定 id，
实现跨会话累积、避免重复确认。这是「概念 id 跨会话稳定」的确定性骨架。

归一化策略（第一版，纯字符串零 token）：
- 全角转半角
- 去空白
- 英文小写
- 去常见泛化后缀（基础/机制/原理/...）

说明：第一版不引入 LLM 额外输出 canonical_name，也不做 embedding 模糊匹配
（两者均列为 Out of Scope），直接从概念名派生稳定匹配键，确定性更强、零额外 token。
"""
from __future__ import annotations

import re
import uuid

import config
import db
import learner_profile

# 常见泛化后缀：去掉后得到「核心词」，用于跨会话匹配。
# 例如「HTTP 基础」「HTTP 协议」都归一到 http。
_GENERIC_SUFFIXES = (
    "基础", "机制", "原理", "概念", "核心", "状态", "队列", "模型",
    "协议", "模式", "结构", "体系", "框架", "理论", "算法", "思想",
    "入门", "进阶", "详解", "实践", "应用", "实战", "指南", "教程",
    "介绍", "总结", "笔记",
)


def _full_to_half(s: str) -> str:
    """全角字符转半角（含全角空格）。"""
    out = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def normalize_key(name: str) -> str:
    """概念名 -> 稳定匹配键（去修饰限定 + 字符串归一）。"""
    if not name:
        return ""
    s = _full_to_half(name.strip())
    s = re.sub(r"\s+", "", s)
    # 循环去除泛化后缀（如「HTTP 协议基础」→ 先去掉「基础」再「协议」）
    changed = True
    while changed:
        changed = False
        for suf in _GENERIC_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf):
                s = s[: -len(suf)]
                changed = True
                break
    return s.lower()


def resolve_global_id(name: str, summary: str = "") -> str:
    """把概念名归一化并 upsert 到全局概念库，返回全局 id。

    命中已有概念则复用其 id（跨会话稳定）；未命中则新建（id = normalize_key）。
    任何异常都安全降级：仍返回一个可用 id，不阻塞主流程。
    """
    key = normalize_key(name)
    if not key:
        # 归一化失败（空名）：退化为带前缀的唯一 id，保证不撞库
        return f"c-{uuid.uuid4().hex[:12]}"
    try:
        db.upsert_concept(key, name, summary)
    except Exception:
        pass  # 落库失败不阻塞，id 仍可返回
    return key


def _state_from_mastery(mastery: float) -> str:
    """掌握度 -> 认知状态（与 cognitive.state_from_mastery 对齐，避免跨模块依赖）。"""
    if mastery >= config.MASTERY_THRESHOLD:
        return "understood"
    if mastery >= config.STATE_BANDS.get("partial", 0.55):
        return "partial"
    return "insufficient"


def build_atlas_view(user_id: str = db.DEFAULT_USER_ID) -> dict:
    """聚合全局概念 + 关系 + 掌握度，返回版图视图。

    - concepts：全局概念库全部概念，附 mastery/state（来自跨会话 concept_mastery，含遗忘衰减）
    - relations：概念关系（is-a/related/prerequisite）
    全局库为空时返回空列表，不报错。
    """
    concepts = [
        {
            "id": c["id"],
            "name": c["name"],
            "summary": c.get("summary", ""),
            "mastery": 0.0,
            "state": "insufficient",
        }
        for c in db.list_concepts()
    ]
    prior = learner_profile.prior_mastery(user_id)
    # 叠加 active 会话实时掌握度（版图随学习实时更新，长期记忆仍保持「完成时沉淀」抗噪语义）
    for cid, m in _live_mastery_from_active_sessions().items():
        prior[cid] = m
    for c in concepts:
        if c["id"] in prior:
            c["mastery"] = round(prior[c["id"]], 4)
            c["state"] = _state_from_mastery(prior[c["id"]])
    relations = [
        {"from": r["from_id"], "to": r["to_id"], "relation_type": r["relation_type"]}
        for r in db.list_concept_relations()
    ]
    mastery_map = {c["id"]: c["mastery"] for c in concepts}
    name_map = {c["id"]: c["name"] for c in concepts}
    bridges = []
    for b in find_bridge_paths([c["id"] for c in concepts], relations, mastery_map):
        bridges.append({
            "bridge": b["bridge"],
            "bridge_name": name_map.get(b["bridge"], b["bridge"]),
            "neighbors": [{"id": n, "name": name_map.get(n, n)} for n in b["neighbors"]],
            "betweenness": b["betweenness"],
        })
    return {"concepts": concepts, "relations": relations, "bridges": bridges}


def neighbors(concept_id: str, depth: int = 1) -> list[dict]:
    """返回某概念的 N 层邻接（BFS），每条含 from/to/relation_type/depth/node。

    孤立概念（无任何邻接）返回空列表，不报错。
    """
    if depth < 1:
        depth = 1
    visited = {concept_id}
    result: list[dict] = []
    frontier = [concept_id]
    for d in range(1, depth + 1):
        nxt: list[str] = []
        for cid in frontier:
            for rel in db.get_concept_relations(cid):
                other = rel["to_id"] if rel["from_id"] == cid else rel["from_id"]
                if other in visited:
                    continue
                visited.add(other)
                nxt.append(other)
                result.append({
                    "from": rel["from_id"],
                    "to": rel["to_id"],
                    "relation_type": rel["relation_type"],
                    "depth": d,
                    "node": other,
                })
        frontier = nxt
    return result


def betweenness_centrality(concept_ids: list[str], relations: list[dict]) -> dict:
    """Brandes 算法计算无权图各节点介数中心性（非归一化）。

    纯 Python 实现，避免引入 networkx 依赖（符合「轻量」决策，
    个人版图规模（数百~数千节点）下毫秒级可完成）。
    """
    adj = {n: [] for n in concept_ids}
    for r in relations:
        f, t = r["from"], r["to"]
        if f in adj and t in adj:
            adj[f].append(t)
            adj[t].append(f)

    c = {n: 0.0 for n in concept_ids}
    for s in concept_ids:
        stack: list[str] = []
        pred = {n: [] for n in concept_ids}
        sigma = {n: 0.0 for n in concept_ids}
        sigma[s] = 1.0
        dist = {n: -1 for n in concept_ids}
        dist[s] = 0
        queue = [s]
        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {n: 0.0 for n in concept_ids}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                c[w] += delta[w]
    return c


def find_bridge_paths(
    concept_ids: list[str],
    relations: list[dict],
    mastery_map: dict,
    threshold: float | None = None,
) -> list[dict]:
    """找桥接通路：桥接节点未掌握，但其两侧（>=2 个）邻居均已掌握。

    返回 [{bridge, neighbors, betweenness}]，无桥接节点时返回空列表（不报错）。
    """
    if threshold is None:
        threshold = config.MASTERY_THRESHOLD
    bc = betweenness_centrality(concept_ids, relations)

    adj = {n: [] for n in concept_ids}
    for r in relations:
        f, t = r["from"], r["to"]
        if f in adj and t in adj:
            adj[f].append(t)
            adj[t].append(f)

    bridges: list[dict] = []
    for cid in concept_ids:
        if mastery_map.get(cid, 0.0) >= threshold:
            continue  # 已掌握，不是桥接点
        mastered_neighbors = [n for n in adj[cid] if mastery_map.get(n, 0.0) >= threshold]
        if len(mastered_neighbors) >= 2:
            bridges.append({
                "bridge": cid,
                "neighbors": mastered_neighbors,
                "betweenness": round(bc.get(cid, 0.0), 4),
            })
    return bridges


def _live_mastery_from_active_sessions() -> dict:
    """汇总当前所有 active 会话的实时掌握度（每个概念取最大值）。

    用于让版图随学习实时更新，而不污染 concept_mastery 的「完成时沉淀」抗噪语义。
    """
    live: dict[str, float] = {}
    for s in db.list_sessions():
        if s.get("status") != "active":
            continue
        session = db.get_session(s["id"])
        cognitive = session.get("cognitive") or {}
        for m in cognitive.get("concepts", []):
            cid = m.get("concept_id")
            if not cid:
                continue
            mastery = float(m.get("mastery", 0.0))
            if mastery > live.get(cid, 0.0):
                live[cid] = mastery
    return live


def build_knowledge_from_concept(concept_id: str) -> dict | None:
    """以全局概念为目标，复用全局库子图构建知识模型（不重新随机拆解）。

    子图 = 目标概念 + 其邻接（prerequisite 前置 / related 相关）。
    返回 KnowledgeModel 结构的 dict；概念不存在返回 None。
    """
    root = db.get_concept(concept_id)
    if root is None:
        return None

    included = {concept_id}
    relations = db.get_concept_relations(concept_id)
    for rel in relations:
        other = rel["to_id"] if rel["from_id"] == concept_id else rel["from_id"]
        included.add(other)

    concepts: list[dict] = []
    for cid in included:
        c = db.get_concept(cid)
        if c is None:
            continue
        concepts.append({
            "id": c["id"],
            "name": c["name"],
            "summary": c.get("summary", ""),
            "why_matters": "",
            "prerequisites": [],
            "common_misconceptions": [],
        })

    # prerequisite 关系：from 是 to 的前置（学 to 前要先学 from）
    by_id = {c["id"]: c for c in concepts}
    for rel in relations:
        if rel["relation_type"] != "prerequisite":
            continue
        from_id, to_id = rel["from_id"], rel["to_id"]
        if to_id in by_id and from_id in by_id:
            by_id[to_id]["prerequisites"].append(from_id)

    return {
        "goal": root["name"],
        "root_concepts": [concept_id],
        "concepts": list(by_id.values()),
    }
