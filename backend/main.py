"""Cognia 后端主应用：FastAPI 入口与学习闭环编排。

核心闭环（对应 Purpose.md）：
学习目标 → 构建知识模型 → 认知诊断 → 更新认知模型 → 教学决策 → 苏格拉底对话
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
import db
import optimizer
import cognitive as cog
import decision
import domain_model
import tutor
from schemas import (
    ChatRequest,
    Concept,
    DiagnosticResult,
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
    """初始化认知模型：按用户先验水平赋予不同的初始掌握概率（千人千面）。"""
    level = cog.infer_prior_level(goal)
    params = cog.get_profile_params(level)
    return {
        "goal": goal,
        "profile": {
            "level": level,
            "label": params["label"],
            "params": {k: v for k, v in params.items() if k != "label"},
        },
        "concepts": [
            {
                "concept_id": c.id,
                "concept_name": c.name,
                "mastery": params["P_L0"],
                "state": "insufficient",
                "evidence_count": 0,
                "consecutive_failures": 0,
                "last_evidence": "",
            }
            for c in concepts
        ],
        "updated_at": _now(),
    }


def _focus_concept_subset(
    focus_id: str | None, concepts: list[Concept]
) -> list[Concept]:
    """只保留焦点概念及其前置概念（含传递闭包），减少诊断 token。

    诊断「用户对当前焦点概念的理解」只需焦点概念及其依赖，无需整张概念图。
    这是「节约」约束的直接落地：砍掉无关概念的冗余信息。
    """
    if focus_id is None:
        return concepts
    by_id = {c.id: c for c in concepts}
    if focus_id not in by_id:
        return concepts
    subset_ids = {focus_id}
    stack = list(by_id[focus_id].prerequisites)
    while stack:
        pid = stack.pop()
        if pid in subset_ids or pid not in by_id:
            continue
        subset_ids.add(pid)
        stack.extend(by_id[pid].prerequisites)
    return [by_id[cid] for cid in subset_ids]


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
    focus = decision.next_focus_concept(knowledge_dict, cognitive)
    intro = tutor.build_intro(knowledge_dict, focus)

    msg = {"role": "assistant", "content": intro, "action": "probe", "diagnosis": None}
    db.append_messages(session["id"], [msg])
    session["messages"] = [msg]
    session["focus_concept"] = focus
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


def _get_active_session(sid: str) -> dict:
    s = db.get_session(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if s["status"] == "completed":
        raise HTTPException(status_code=400, detail="该学习目标已完成")
    return s


def _process_turn(s: dict, content: str) -> dict:
    """执行「诊断 → 推进意图路由 → 认知更新 → 教学决策」，返回中间结果 ctx。

    不含回复生成与持久化，供非流式 chat 与流式 chat_stream 共用。
    """
    knowledge = s["knowledge"]
    cognitive = s["cognitive"]
    concepts = [Concept(**c) for c in knowledge["concepts"]]
    trace: list[dict] = []  # 本轮「思考过程」轨迹（各层 LLM 调用的 prompt 与原始输出）

    # 1. 确定当前焦点概念
    focus = decision.next_focus_concept(knowledge, cognitive, trace=trace)
    focus_id = focus["id"] if focus else None

    # 2. 认知诊断（只传焦点概念子集，减少 token）
    focus_concepts = _focus_concept_subset(focus_id, concepts)
    diagnosis = cog.diagnose(knowledge["goal"], focus_concepts, content, focus_id, trace=trace)

    # 2.5 推进意图路由（确定性规则，零 token）：用户明确要求「继续/下一个」时，
    #     预标记当前焦点概念为已掌握并覆盖诊断为 understood，让后续焦点选择
    #     自然推进到下一个概念，避免被误判 insufficient 而陷入反复解释的死循环。
    advance_intent = decision.detect_advance_intent(content)
    if advance_intent and focus_id:
        for m in cognitive["concepts"]:
            if m["concept_id"] == focus_id:
                m["mastery"] = config.MASTERY_THRESHOLD + 0.05
                m["state"] = "understood"
                break
        diagnosis = DiagnosticResult(
            state="understood",
            confidence=0.85,
            concept_ids=[focus_id],
            evidence=content[:60],
            misconception="",
            missing=[],
        )

    # 3. 更新认知模型
    mastery_map = {m["concept_id"]: m for m in cognitive["concepts"]}
    target_ids = diagnosis.concept_ids or ([focus_id] if focus_id else [])
    if not target_ids:
        target_ids = [c.id for c in concepts]

    profile_params = (cognitive.get("profile") or {}).get("params") or {}
    is_failure = diagnosis.state in ("misconceived", "insufficient")
    for cid in target_ids:
        m = mastery_map.get(cid)
        if m is None:
            continue
        new_mastery = cog.bayes_update(
            m["mastery"],
            diagnosis.state,
            diagnosis.confidence,
            slip=profile_params.get("P_SLIP"),
            guess=profile_params.get("P_GUESS"),
            learn=profile_params.get("P_LEARN"),
        )
        m["mastery"] = round(new_mastery, 4)
        m["evidence_count"] += 1
        m["last_evidence"] = diagnosis.evidence or content[:60]
        m["state"] = cog.state_from_mastery(new_mastery)
        if is_failure:
            m["consecutive_failures"] = m.get("consecutive_failures", 0) + 1
        else:
            m["consecutive_failures"] = 0

    # 若诊断为错误，即使掌握概率不低也要显式标注，便于前端区分
    if diagnosis.state == "misconceived":
        for cid in target_ids:
            m = mastery_map.get(cid)
            if m:
                m["state"] = "misconceived"

    cognitive["updated_at"] = _now()

    # 4. 教学决策（分层流程控制：候选集内 LLM 决策 + 规则回退）
    focus_mastery = mastery_map.get(focus_id, {}) if focus_id else {}
    decision_result = decision.decide_action(
        diagnosis.state,
        focus_mastery.get("evidence_count", 0),
        consecutive_failures=focus_mastery.get("consecutive_failures", 0),
        concept_name=focus["name"] if focus else "",
        diagnosis=diagnosis,
        mastery=focus_mastery.get("mastery", 0.0),
        trace=trace,
    )

    return {
        "content": content,
        "knowledge": knowledge,
        "cognitive": cognitive,
        "focus": focus,
        "focus_id": focus_id,
        "diagnosis": diagnosis,
        "decision_result": decision_result,
        "action": decision_result.chosen_action,
        "completed": _all_root_mastered(knowledge, cognitive),
        "advance_intent": advance_intent,
        "trace": trace,
    }


def _resolve_reply(ctx: dict) -> tuple[str | None, str, str, bool]:
    """根据 ctx 决定回复文本、action、status、是否需要流式生成。

    返回 (reply, action, status, is_stream)：
    - completed / 无焦点 / 推进意图：确定性文本，is_stream=False
    - 正常 tutor 回复：reply=None，is_stream=True（由调用方流式生成）
    """
    if ctx["completed"] or ctx["focus"] is None:
        return _complete_reply(), "advance", "completed", False

    if ctx["advance_intent"]:
        next_focus = decision.next_focus_concept(ctx["knowledge"], ctx["cognitive"], trace=ctx["trace"])
        if next_focus is not None:
            ctx["focus"] = next_focus
            reply = (
                f"好的，这个点你已经掌握了。我们接着看下一个概念：**{next_focus['name']}**。"
                f"\n\n在讲解之前，先听听你的理解——你能用自己的话说说，"
                f"「{next_focus['name']}」是什么、解决什么问题吗？"
            )
            return reply, "advance", "active", False
        return _complete_reply(), "advance", "completed", False

    return None, ctx["action"], "active", True


def _persist_user_message(sid: str, content: str) -> None:
    """流程开始即落库用户消息（诊断稍后回填），避免流式回复期间刷新丢消息。"""
    db.append_user_message(sid, content)


def _persist(sid: str, ctx: dict, reply: str, action: str, status: str) -> dict:
    """完成决策统一、回填用户诊断、追加 AI 回复并持久化，返回响应 payload。"""
    decision_result = ctx["decision_result"]
    # 完成时统一决策语义为「完成推进」，保证 decision 与 action 一致
    if status == "completed":
        decision_result = decision.ActionDecision(
            chosen_action="advance",
            reasons=decision.ActionReason(
                criterion_used="完成判定",
                pedagogical_intent="学习目标已完成",
                confidence=1.0,
            ),
        )
        action = "advance"

    # 用户消息已在流程开始时立即落库（diagnosis 暂空），此处回填诊断结果
    db.update_last_user_diagnosis(sid, ctx["diagnosis"].model_dump())
    ai_msg = {"role": "assistant", "content": reply, "action": action, "diagnosis": ctx["diagnosis"].model_dump(), "decision": decision_result.model_dump(), "trace": ctx["trace"]}
    db.append_messages(sid, [ai_msg])
    db.update_session(sid, ctx["cognitive"], ctx["knowledge"], status=status)

    return {
        "session_id": sid,
        "action": action,
        "reply": reply,
        "diagnosis": ctx["diagnosis"].model_dump(),
        "decision": decision_result.model_dump(),
        "cognitive": ctx["cognitive"],
        "status": status,
        "focus_concept": ctx["focus"],
        "trace": ctx["trace"],
    }


@app.post("/api/sessions/{sid}/chat")
def chat(sid: str, req: ChatRequest):
    s = _get_active_session(sid)
    _persist_user_message(sid, req.content)  # 立即落库用户消息，防刷新丢失
    ctx = _process_turn(s, req.content)

    reply, action, status, is_stream = _resolve_reply(ctx)
    if is_stream:
        reply = tutor.generate_tutor_reply(
            Concept(**ctx["focus"]), ctx["diagnosis"], ctx["action"], req.content, trace=ctx["trace"]
        )

    return _persist(sid, ctx, reply, action, status)


@app.post("/api/sessions/{sid}/chat/stream")
def chat_stream(sid: str, req: ChatRequest):
    s = _get_active_session(sid)
    _persist_user_message(sid, req.content)  # 立即落库用户消息，防刷新丢失
    ctx = _process_turn(s, req.content)
    reply, action, status, is_stream = _resolve_reply(ctx)

    def event_stream():
        if is_stream:
            parts: list[str] = []
            for delta in tutor.stream_tutor_reply(
                Concept(**ctx["focus"]), ctx["diagnosis"], ctx["action"], req.content, trace=ctx["trace"]
            ):
                parts.append(delta)
                yield f"data: {json.dumps({'type': 'token', 'content': delta}, ensure_ascii=False)}\n\n"
            reply_final = "".join(parts).strip()
        else:
            reply_final = reply or ""

        payload = _persist(sid, ctx, reply_final, action, status)
        yield f"data: {json.dumps({'type': 'done', 'data': payload}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.delete("/api/sessions/{sid}")
def delete_session(sid: str):
    if not db.delete_session(sid):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}

@app.delete("/api/sessions/{sid}/messages/{index}")
def delete_message(sid: str, index: int):
    messages = db.delete_message(sid, index)
    if messages is None:
        raise HTTPException(status_code=404, detail="消息不存在")
    return {"ok": True, "messages": messages}

@app.put("/api/sessions/{sid}/messages/{index}")
def update_message(sid: str, index: int, req: ChatRequest):
    messages = db.update_message(sid, index, req.content)
    if messages is None:
        raise HTTPException(status_code=404, detail="消息不存在")
    return {"ok": True, "messages": messages}

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


_optimizer_task: asyncio.Task | None = None


@app.on_event("startup")
async def startup():
    db.init_db()
    # 启动在线 Prompt 优化反馈回路（慢循环）
    if config.OPTIMIZER_ENABLED and config.AI_ENABLED:
        global _optimizer_task
        _optimizer_task = asyncio.create_task(optimizer.optimize_loop())


@app.on_event("shutdown")
async def shutdown():
    global _optimizer_task
    if _optimizer_task is not None:
        _optimizer_task.cancel()
        try:
            await _optimizer_task
        except asyncio.CancelledError:
            pass
        _optimizer_task = None
