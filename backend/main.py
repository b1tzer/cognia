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
import atlas
import cognitive as cog
import decision
import domain_model
import goal_clarify
import learner_profile
import tutor
from schemas import (
    ChatRequest,
    Concept,
    DiagnosticResult,
    KnowledgeModel,
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
    """初始化认知模型：按用户先验水平 + 跨会话历史掌握度赋予初始掌握概率。

    千人千面：优先用 learner_profile 的历史掌握度做先验（含遗忘衰减），
    无历史的概念回退到按目标推断的先验水平 P_L0。
    """
    level = cog.infer_prior_level(goal)
    params = cog.get_profile_params(level)
    prior = learner_profile.prior_mastery()  # {concept_id: 遗忘衰减后的 mastery}
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
                "mastery": prior.get(c.id, params["P_L0"]),
                "state": "insufficient",
                "evidence_count": 0,
                "consecutive_failures": 0,
                "last_evidence": "",
                "success_count": 0,
                "quality": "",
                "mastered": False,
                "dialogue": [],
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

# 对话栈（概念级短期记忆）：跟着焦点概念走，切换焦点即切栈，token 天然有界
_DIALOGUE_MAX_DEPTH = 3      # 每个概念最多保留的轨迹轮数
_DIALOGUE_USER_MAX = 200     # user_text 截断长度（字符）
_DIALOGUE_AI_MAX = 120       # ai_reply 截断长度（字符）

def _focus_history(cognitive: dict, focus_id: str | None) -> list:
    """返回焦点概念的 dialogue 栈；无焦点或无栈时返回空列表。"""
    if not focus_id:
        return []
    for m in cognitive.get("concepts", []):
        if m.get("concept_id") == focus_id:
            return m.get("dialogue", [])
    return []

def _push_dialogue(
    cognitive: dict,
    focus_id: str | None,
    user_text: str,
    ai_reply: str,
    action: str,
    state: str,
) -> None:
    """把本轮轨迹压入焦点概念的 dialogue 栈（容量上限 N，字段按长度截断）。

    只保留最近 _DIALOGUE_MAX_DEPTH 轮，超出丢最旧。focus_id 为空时跳过。
    """
    if not focus_id:
        return
    for m in cognitive.get("concepts", []):
        if m.get("concept_id") == focus_id:
            stack = m.setdefault("dialogue", [])
            stack.append({
                "user_text": (user_text or "")[:_DIALOGUE_USER_MAX],
                "ai_reply": (ai_reply or "")[:_DIALOGUE_AI_MAX],
                "action": action,
                "state": state,
            })
            del stack[: max(0, len(stack) - _DIALOGUE_MAX_DEPTH)]
            break

def _all_root_mastered(knowledge: dict, cognitive: dict) -> bool:
    mastery = {m["concept_id"]: m for m in cognitive["concepts"]}
    roots = knowledge.get("root_concepts", [])
    if not roots:
        return False
    return all(decision.is_mastered(mastery.get(r, {})) for r in roots)


def _complete_reply() -> str:
    return (
        "🎉 恭喜！你已经掌握了这个学习目标下的核心概念，能够准确表达理解、"
        "并能在新场景中迁移应用。\n\n"
        "回顾一下我们走过的路径：从基础概念出发，逐步串起因果链，纠正了误解，"
        "最终形成了完整、正确、可迁移的理解。\n\n"
        "如果你愿意，可以设定一个新的学习目标，我们继续。"
    )


def _clarify_session(sid: str, goal: str, clarify: dict) -> dict:
    """进入澄清阶段：不建知识模型，追加澄清提问消息，等待用户确认目标。"""
    db.update_goal(sid, goal, None, None, stage="clarifying")
    msg = {
        "role": "assistant",
        "content": clarify["question"],
        "action": None,
        "diagnosis": None,
        "clarify": {"candidates": clarify["candidates"], "question": clarify["question"]},
    }
    db.append_messages(sid, [msg])
    s = db.get_session(sid)
    s["focus_concept"] = None
    return s


def _finalize_with_knowledge(sid: str, goal: str, knowledge: KnowledgeModel) -> dict:
    """用给定知识模型构建认知模型，切回 active 并追加开场白（供目标/概念两条路径复用）。"""
    cognitive = _build_cognitive(goal, knowledge.concepts)
    knowledge_dict = knowledge.model_dump()
    focus = decision.next_focus_concept(knowledge_dict, cognitive)
    intro = tutor.build_intro(knowledge_dict, focus)
    db.update_goal(sid, goal, knowledge_dict, cognitive, stage="active")
    msg = {"role": "assistant", "content": intro, "action": "probe", "diagnosis": None}
    db.append_messages(sid, [msg])
    s = db.get_session(sid)
    s["focus_concept"] = focus
    return s


def _finalize_goal(sid: str, goal: str) -> dict:
    """用确定的目标构建知识模型与认知模型，切回 active 并追加开场白。"""
    knowledge = domain_model.build_knowledge_model(goal)
    return _finalize_with_knowledge(sid, goal, knowledge)


def _handle_goal_change(sid: str, raw_text: str) -> dict:
    """处理「中途修改目标」：让 LLM 从自然语言中规范化出真实目标，走澄清/重建。

    返回与 _persist 对齐的响应 payload，额外携带 stage/clarify 供前端判断是否重载。
    """
    clarify = goal_clarify.clarify_goal(raw_text)
    if clarify["need_clarify"]:
        s = _clarify_session(sid, clarify["goal"], clarify)
    else:
        s = _finalize_goal(sid, clarify["goal"])

    last_msg = s["messages"][-1] if s["messages"] else {}
    return {
        "session_id": sid,
        "action": "probe",
        "reply": last_msg.get("content", ""),
        "diagnosis": None,
        "decision": None,
        "cognitive": s.get("cognitive"),
        "status": s.get("status"),
        "stage": s.get("stage"),
        "focus_concept": s.get("focus_concept"),
        "trace": [],
        "clarify": last_msg.get("clarify"),
    }


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True, "ai_enabled": config.AI_ENABLED, "model": config.OPENAI_MODEL if config.AI_ENABLED else None}

@app.get("/api/atlas")
def get_atlas():
    """全局个人知识版图：返回全局概念 + 关系 + 掌握度聚合。"""
    return atlas.build_atlas_view()

@app.get("/api/atlas/{concept_id}/neighbors")
def get_neighbors(concept_id: str, depth: int = 1):
    """概念周边关联：返回 N 层邻接（is-a/related/prerequisite）。"""
    return {"concept_id": concept_id, "neighbors": atlas.neighbors(concept_id, depth)}

@app.post("/api/atlas/{concept_id}/session")
def start_concept_session(concept_id: str):
    """点击版图概念发起学习会话：复用全局库子图构建知识模型，不重新随机拆解。"""
    km_dict = atlas.build_knowledge_from_concept(concept_id)
    if km_dict is None:
        raise HTTPException(status_code=404, detail="概念不存在")
    knowledge = KnowledgeModel(**km_dict)
    session = db.create_session(knowledge.goal, stage="active")
    return _finalize_with_knowledge(session["id"], knowledge.goal, knowledge)


@app.post("/api/sessions")
def start_session(req: StartSessionRequest):
    goal = req.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="学习目标不能为空")

    # 需求澄清：有歧义/同名概念时先确认，再构建知识模型
    clarify = goal_clarify.clarify_goal(goal)
    if clarify["need_clarify"]:
        session = db.create_session(goal, stage="clarifying")
        return _clarify_session(session["id"], goal, clarify)

    session = db.create_session(goal, stage="active")
    return _finalize_goal(session["id"], goal)


@app.post("/api/sessions/{sid}/confirm-goal")
def confirm_goal(sid: str, req: StartSessionRequest):
    """澄清阶段：用户确认目标（点选候选或输入自定义）后，构建知识模型正式开始。"""
    s = db.get_session(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    goal = req.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="学习目标不能为空")
    return _finalize_goal(sid, goal)


@app.put("/api/sessions/{sid}/goal")
def update_goal(sid: str, req: StartSessionRequest):
    """中途修改目标：重建知识模型与认知模型，目标有歧义则先转澄清。"""
    s = db.get_session(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    goal = req.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="学习目标不能为空")

    clarify = goal_clarify.clarify_goal(goal)
    if clarify["need_clarify"]:
        return _clarify_session(sid, goal, clarify)
    return _finalize_goal(sid, goal)


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


def _apply_evidence_fields(m: dict, diagnosis: DiagnosticResult) -> None:
    """更新 L2 证据层 success_count 与 L3 理解质量层 quality。

    - success_count：understood +1 / misconceived 重置 0 / 其余不变
    - quality：仅 understood 记录 deep/surface，其余置空
    """
    if diagnosis.state == "understood":
        m["success_count"] = m.get("success_count", 0) + 1
    elif diagnosis.state == "misconceived":
        m["success_count"] = 0
    if diagnosis.state == "understood":
        m["quality"] = diagnosis.quality
    else:
        m["quality"] = ""


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

    # 1.5 提前取焦点认知快照（供合并决策的停滞检测与步数护栏）
    focus_before = (
        next((m for m in cognitive["concepts"] if m["concept_id"] == focus_id), {})
        if focus_id else {}
    )
    focus_evidence = focus_before.get("evidence_count", 0)
    focus_failures = focus_before.get("consecutive_failures", 0)

    # 2. 合并「认知诊断 + 教学动作决策」为一次 LLM 调用（Map+Guide 单循环）。
    #    替代原 diagnose + decide_action 两次调用，消除 state→action 跨调用契约断裂。
    focus_concepts = _focus_concept_subset(focus_id, concepts)
    focus_history = _focus_history(cognitive, focus_id)
    misconceptions = learner_profile.get_misconceptions_for([c.id for c in focus_concepts])
    diagnosis, action_decision = cog.diagnose_and_decide(
        knowledge["goal"], focus_concepts, content, focus_id,
        evidence_count=focus_evidence,
        consecutive_failures=focus_failures,
        trace=trace, history=focus_history, misconceptions=misconceptions,
    )

    # 2.1 若诊断出误解，沉淀到跨会话长期记忆（学习者画像）
    if diagnosis.state == "misconceived" and diagnosis.misconception and focus_id:
        learner_profile.record_misconception(focus_id, diagnosis.misconception, diagnosis.confidence)

    # 2.5 推进判定（Map+Guide 单循环核心）：推进不再由多层数值 AND 反推，
    #     而是「LLM 语义（understood）＋ 两条确定性护栏（用户推进意图 / 步数上限）」
    #     直接决定 should_advance。推进是默认方向，纠缠是例外。
    advance_intent = decision.detect_advance_intent(content)
    # 步数护栏：更新前的 evidence_count（即「这是第几轮回答」）达上限则强制推进
    steps_over_limit = focus_evidence >= config.MAX_STEPS_PER_CONCEPT
    should_advance = (
        advance_intent                       # 护栏 1：用户明确说「继续/下一个/懂了」
        or steps_over_limit                  # 护栏 2：1~2 次验证后仍推进
        or diagnosis.state == "understood"   # LLM 语义：已判定理解
    )

    # 2.6 若推进，预标记当前焦点为已掌握（覆盖诊断为 understood），
    #     使后续焦点选择自然推进到下一个概念。
    if should_advance and focus_id:
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
        _apply_evidence_fields(m, diagnosis)
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

    # 3.5 若推进，把当前焦点标记为 mastered（推进的权威标记，供焦点选择使用）
    if should_advance and focus_id:
        for m in cognitive["concepts"]:
            if m["concept_id"] == focus_id:
                m["mastered"] = True
                break

    cognitive["updated_at"] = _now()

    # 4. 教学决策：动作已在上面由 diagnose_and_decide 合并产出；
    #    推进时覆盖为 advance（切换焦点由 _resolve_reply 完成）。
    if should_advance:
        decision_result = decision.ActionDecision(
            chosen_action="advance",
            reasons=decision.ActionReason(
                evidence_cited=(
                    "用户明确推进意图" if advance_intent
                    else "步数上限" if steps_over_limit
                    else "诊断判定 understood"
                ),
                criterion_used="推进判定（Map+Guide）",
                pedagogical_intent="当前概念已完成，推进到下一个概念",
                confidence=1.0,
            ),
        )
    else:
        decision_result = action_decision

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
        "should_advance": should_advance,
        "history": focus_history,
        "trace": trace,
    }


def _resolve_reply(ctx: dict) -> tuple[str | None, str, str, bool]:
    """根据 ctx 决定回复文本、action、status、是否需要流式生成。

    返回 (reply, action, status, is_stream)：
    - completed / 无焦点 / 推进判定：确定性文本，is_stream=False
    - 正常 tutor 回复：reply=None，is_stream=True（由调用方流式生成）
    """
    if ctx["completed"] or ctx["focus"] is None:
        return _complete_reply(), "advance", "completed", False

    if ctx["should_advance"]:
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
        # 目标完成：把本目标的各概念掌握度沉淀到跨会话长期记忆（学习者画像）
        learner_profile.persist_concept_mastery(ctx["cognitive"])

    # 用户消息已在流程开始时立即落库（diagnosis 暂空），此处回填诊断结果
    db.update_last_user_diagnosis(sid, ctx["diagnosis"].model_dump())
    ai_msg = {"role": "assistant", "content": reply, "action": action, "decision": decision_result.model_dump(), "trace": ctx["trace"]}
    db.append_messages(sid, [ai_msg])
    # 本轮结束后，把轨迹压入焦点概念的对话栈（供下一轮诊断/回复关联上下文）
    _push_dialogue(
        ctx["cognitive"],
        ctx.get("focus_id"),
        ctx.get("content", ""),
        reply,
        action,
        ctx["diagnosis"].state,
    )
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
    s = db.get_session(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 澄清阶段或改目标意图 → 目标处理分支（重建/继续澄清）
    if s.get("stage") == "clarifying" or goal_clarify.detect_goal_change_intent(req.content):
        _persist_user_message(sid, req.content)
        return _handle_goal_change(sid, req.content)

    if s["status"] == "completed":
        raise HTTPException(status_code=400, detail="该学习目标已完成")

    _persist_user_message(sid, req.content)  # 立即落库用户消息，防刷新丢失
    ctx = _process_turn(s, req.content)

    reply, action, status, is_stream = _resolve_reply(ctx)
    if is_stream:
        reply = tutor.generate_tutor_reply(
            Concept(**ctx["focus"]), ctx["diagnosis"], ctx["action"], req.content,
            trace=ctx["trace"], history=ctx["history"],
        )

    return _persist(sid, ctx, reply, action, status)


@app.post("/api/sessions/{sid}/chat/stream")
def chat_stream(sid: str, req: ChatRequest):
    s = db.get_session(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 澄清阶段或改目标意图 → 目标处理分支（返回简单 SSE 流，done 携带 stage/clarify）
    if s.get("stage") == "clarifying" or goal_clarify.detect_goal_change_intent(req.content):
        _persist_user_message(sid, req.content)
        payload = _handle_goal_change(sid, req.content)

        def goal_stream():
            yield f"data: {json.dumps({'type': 'done', 'data': payload}, ensure_ascii=False)}\n\n"

        return StreamingResponse(goal_stream(), media_type="text/event-stream")

    if s["status"] == "completed":
        raise HTTPException(status_code=400, detail="该学习目标已完成")

    _persist_user_message(sid, req.content)  # 立即落库用户消息，防刷新丢失
    ctx = _process_turn(s, req.content)
    reply, action, status, is_stream = _resolve_reply(ctx)

    def event_stream():
        # 先推送「思考过程」轨迹（焦点选择/认知诊断/教学决策等各层 LLM 调用），
        # 让用户在回答输出之前就能实时看到产品是如何一步步思考的。
        for i, step in enumerate(ctx["trace"]):
            yield f"data: {json.dumps({'type': 'trace', 'step': step, 'index': i}, ensure_ascii=False)}\n\n"

        if is_stream:
            parts: list[str] = []
            for delta in tutor.stream_tutor_reply(
                Concept(**ctx["focus"]), ctx["diagnosis"], ctx["action"], req.content,
                trace=ctx["trace"], history=ctx["history"],
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


@app.post("/api/sessions/{sid}/messages/{index}/regenerate")
def regenerate_message(sid: str, index: int):
    """重新生成某条 AI 回答：基于相同诊断与上下文重新采样回复文本。

    只替换回复措辞，不改动诊断/决策/认知状态（BKT 不重复累计 evidence），
    符合「重新生成 = 相同 prompt 重新采样」的行业语义（Cloudscape / Vercel AI SDK）。
    """
    s = db.get_session(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    if s["status"] == "completed":
        raise HTTPException(status_code=400, detail="该学习目标已完成")

    messages = s["messages"]
    if index < 0 or index >= len(messages) or messages[index].get("role") != "assistant":
        raise HTTPException(status_code=400, detail="只能重新生成 AI 回答")

    target = messages[index]
    diagnosis_data = target.get("diagnosis")
    if not diagnosis_data:
        raise HTTPException(status_code=400, detail="该回答不支持重新生成")
    action = target.get("action") or "probe"

    # 该回答对应的最近一条用户输入
    user_text = ""
    for m in reversed(messages[:index]):
        if m.get("role") == "user":
            user_text = m.get("content", "")
            break

    # 定位焦点概念（优先诊断命中的概念，回退到当前焦点 / 首个概念）
    knowledge = s["knowledge"] or {"concepts": []}
    by_id = {c["id"]: c for c in knowledge.get("concepts", [])}
    focus = None
    for cid in diagnosis_data.get("concept_ids", []):
        if cid in by_id:
            focus = Concept(**by_id[cid])
            break
    if focus is None and s.get("focus_concept"):
        focus = Concept(**s["focus_concept"])
    if focus is None and by_id:
        focus = Concept(**next(iter(by_id.values())))

    diagnosis = DiagnosticResult(**diagnosis_data)
    reply = tutor.generate_tutor_reply(focus, diagnosis, action, user_text)

    if db.update_message(sid, index, reply) is None:
        raise HTTPException(status_code=404, detail="消息不存在")

    return db.get_session(sid)

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
