"""Cognia 后端主应用：FastAPI 入口与学习闭环编排。

核心闭环（对应 Purpose.md）：
学习目标 → 构建知识模型 → 认知诊断 → 更新认知模型 → 教学决策 → 苏格拉底对话
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import config
import db
import cognitive as cog
import domain_model
import tutor
from schemas import (
    ChatRequest,
    Concept,
    StartSessionRequest,
)

app = FastAPI(title="Cognia", description="AI Learning Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_cognitive(goal: str, concepts: list[Concept]) -> dict:
    """初始化认知模型：每个概念赋予初始掌握概率。"""
    return {
        "goal": goal,
        "concepts": [
            {
                "concept_id": c.id,
                "concept_name": c.name,
                "mastery": config.P_L0,
                "state": "insufficient",
                "evidence_count": 0,
                "last_evidence": "",
            }
            for c in concepts
        ],
        "updated_at": _now(),
    }


def _all_root_mastered(knowledge: dict, cognitive: dict) -> bool:
    mastery = {m["concept_id"]: m["mastery"] for m in cognitive["concepts"]}
    roots = knowledge.get("root_concepts", [])
    if not roots:
        return False
    return all(mastery.get(r, 0.0) >= config.MASTERY_THRESHOLD for r in roots)


def _complete_reply() -> str:
    return (
        "🎉 恭喜！你已经掌握了这个学习目标下的核心概念，能够准确表达理解、"
        "并能在新场景中迁移应用。\n\n"
        "回顾一下我们走过的路径：从基础概念出发，逐步串起因果链，纠正了误解，"
        "最终形成了完整、正确、可迁移的理解。\n\n"
        "如果你愿意，可以设定一个新的学习目标，我们继续。"
    )


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "ai_enabled": config.AI_ENABLED, "model": config.OPENAI_MODEL if config.AI_ENABLED else None}


@app.post("/api/sessions")
def start_session(req: StartSessionRequest):
    goal = req.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="学习目标不能为空")

    knowledge = domain_model.build_knowledge_model(goal)
    cognitive = _build_cognitive(goal, knowledge.concepts)
    knowledge_dict = knowledge.model_dump()

    session = db.create_session(goal, knowledge_dict, cognitive)
    focus = tutor.next_focus_concept(knowledge_dict, cognitive)
    intro = tutor.build_intro(knowledge_dict, focus)

    msg = {"role": "assistant", "content": intro, "action": "probe", "diagnosis": None}
    db.append_messages(session["id"], [msg])
    session["messages"] = [msg]
    return session


@app.get("/api/sessions")
def list_sessions():
    return db.list_sessions()


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    s = db.get_session(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return s


@app.post("/api/sessions/{sid}/chat")
def chat(sid: str, req: ChatRequest):
    s = db.get_session(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if s["status"] == "completed":
        raise HTTPException(status_code=400, detail="该学习目标已完成")

    knowledge = s["knowledge"]
    cognitive = s["cognitive"]
    concepts = [Concept(**c) for c in knowledge["concepts"]]

    # 1. 确定当前焦点概念
    focus = tutor.next_focus_concept(knowledge, cognitive)
    focus_id = focus["id"] if focus else None

    # 2. 认知诊断
    diagnosis = cog.diagnose(knowledge["goal"], concepts, req.content, focus_id)

    # 3. 更新认知模型
    mastery_map = {m["concept_id"]: m for m in cognitive["concepts"]}
    # 无焦点或未识别到概念时，将证据归到焦点概念（或全部相关概念）
    target_ids = diagnosis.concept_ids or ([focus_id] if focus_id else [])
    if not target_ids:
        target_ids = [c.id for c in concepts]

    for cid in target_ids:
        m = mastery_map.get(cid)
        if m is None:
            continue
        new_mastery = cog.bayes_update(m["mastery"], diagnosis.state, diagnosis.confidence)
        m["mastery"] = round(new_mastery, 4)
        m["evidence_count"] += 1
        m["last_evidence"] = diagnosis.evidence or req.content[:60]
        m["state"] = cog.state_from_mastery(new_mastery)

    # 若诊断为错误，即使掌握概率不低也要显式标注，便于前端区分
    if diagnosis.state == "misconceived":
        for cid in target_ids:
            m = mastery_map.get(cid)
            if m:
                m["state"] = "misconceived"

    cognitive["updated_at"] = _now()

    # 4. 教学决策 + 生成回复
    action = tutor.decide_action(diagnosis.state, mastery_map.get(focus_id, {}).get("evidence_count", 0) if focus_id else 0)
    completed = _all_root_mastered(knowledge, cognitive)

    if completed:
        reply = _complete_reply()
        status = "completed"
        action = "advance"
    else:
        if focus is None:
            reply = _complete_reply()
            status = "completed"
            action = "advance"
        else:
            reply = tutor.generate_tutor_reply(Concept(**focus), diagnosis)
            status = "active"

    # 5. 持久化
    user_msg = {"role": "user", "content": req.content, "action": None, "diagnosis": diagnosis.model_dump()}
    ai_msg = {"role": "assistant", "content": reply, "action": action, "diagnosis": diagnosis.model_dump()}
    db.append_messages(sid, [user_msg, ai_msg])
    db.update_session(sid, cognitive, knowledge, status=status)

    return {
        "session_id": sid,
        "action": action,
        "reply": reply,
        "diagnosis": diagnosis.model_dump(),
        "cognitive": cognitive,
        "status": status,
        "focus_concept": focus,
    }


# ---------------------------------------------------------------------------
# 前端静态托管（构建产物）
# ---------------------------------------------------------------------------
DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        candidate = DIST / full_path
        if full_path and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")


@app.on_event("startup")
def startup():
    db.init_db()
