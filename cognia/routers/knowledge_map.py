"""知识版图聚合路由（/knowledge-map）。

聚合当前匿名用户的全局知识星图（跨对话概念合并 + 熟练度），供个人页渲染。
与 AG-UI 接入层解耦，属纯查询端点。
"""

from fastapi import APIRouter, HTTPException, Request

from cognia import memory

router = APIRouter()


@router.get("/knowledge-map")
async def knowledge_map_endpoint(request: Request, user_id: str | None = None):
    """聚合当前匿名用户的全局知识星图（跨对话概念合并 + 熟练度），供个人页渲染。

    - `user_id` 缺失 → 400（前端必须先建立匿名标识）。
    - store 为 None（Postgres 降级）→ 返回空版图 `goals: []`，不 500。
    - 以 user 为单位：跨 goal 按 point_id（全局稳定 id）去重合并为**单张全局图**，
      同一概念只保留一个节点，依赖边去重，熟练度按全局 id 对齐（缺失补 unassessed）。
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="缺少 user_id 参数")

    store = getattr(request.app.state, "store", None)
    if store is None:
        return {"user_id": user_id, "goals": []}

    kms = await memory.alist_knowledge_models(store, user_id)
    proficiencies = await memory.alist_authoritative_proficiencies(store, user_id)

    # 跨 goal 聚合：按全局 point_id 去重合并（同一概念只保留一个节点）
    points_by_id: dict[str, dict] = {}
    for km in kms:
        for p in (km.get("points") or []):
            pid = p.get("id")
            if not pid:
                continue
            if pid not in points_by_id:
                points_by_id[pid] = {
                    "id": pid,
                    "name": p.get("name"),
                    "description": p.get("description"),
                    "prerequisites": [],
                }
            # 合并依赖边（去重）
            for pre in (p.get("prerequisites") or []):
                if pre not in points_by_id[pid]["prerequisites"]:
                    points_by_id[pid]["prerequisites"].append(pre)

    points = list(points_by_id.values())
    global_proficiencies = {
        pid: proficiencies.get(pid, "unassessed") for pid in points_by_id
    }

    return {
        "user_id": user_id,
        "goals": [{
            "goal": "我的知识星图",
            "points": points,
            "proficiencies": global_proficiencies,
        }],
    }
