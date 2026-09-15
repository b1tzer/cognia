"""Cognia 编排层（LangGraph）：Agent 循环，而非固定流水线。

架构原则（本次重构核心）：
- **流程控制学习，Agent 控制对话**：图不再设计成
  `收目标 → 拆知识点 → 出题 → 等回答 → 诊断 → 验证 → 讲解 → 下一步` 的固定流水线，
  而是一个「对话 Agent ⇄ 学习引擎」的动态循环。
- Conversation Agent（conversation_agent.py）负责理解用户、决定如何回应与教学，
  拥有**对话自主权**。
- Learning Engine（learning_engine.py）负责学习事实与合法状态变更，拥有**认知状态
  最终裁决权**（经诊断 → 验证 → 状态机）。
- State Machine（state_machine.py）只做「裁判」：约束哪些认知状态迁移合法、
  掌握必须经过双重验证，不让 LLM 随意改长期认知状态。

图结构（单循环 + HITL 中断）：

    START → conversation_agent
      ├─ action ∈ {diagnose, confirm_mastery, set_goal, change_topic}
      │        → learning_engine → conversation_agent（生成最终回复）
      ├─ action == end → END
      └─ 其余（respond/clarify/probe/explain/follow_up）
               → await_user(interrupt) → conversation_agent

其中 learning_engine 执行完把 turn_phase 置为 "reply"，让 Agent 基于引擎结果生成
最终回复；await_user 用 interrupt 暂停，把回复交给用户并等待下一轮输入。
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from cognia import models
from cognia.conversation_agent import (
    ConversationAction,
    ConversationTurn,
    build_agent_messages,
)
from cognia.learning_engine import (
    _dump,
    _load_knowledge_model,
    _load_state,
    _load_verification,
    apply_migration,
    build_knowledge_model,
    improved,
    read_historical_state,
    resolve_migration,
    run_diagnosis,
    run_verification,
)
from cognia.memory import (
    get_knowledge_model,
    get_profile_dict,
    get_user_id,
    normalize_goal,
    put_knowledge_model,
)
from cognia.schemas import (
    CognitiveState,
    Confidence,
    VerificationState,
)
from cognia.state_machine import is_mastered_migration_allowed

# ---- 安全上限（spec §6 硬约束：防失控 / 防烧钱）----

MAX_INTERVENTION_FAILS = 3  # 单知识点干预失败轮次上限，达此值触发放弃（回溯）
MAX_PROBE_STALLS = 3         # 连续低/中置信度无进展轮次上限，达此值放弃该知识点

# 需要交给学习引擎处理的 Agent 动作（其余为纯对话动作，直接面向用户）
ENGINE_ACTIONS = {
    ConversationAction.DIAGNOSE,
    ConversationAction.CONFIRM_MASTERY,
    ConversationAction.SET_GOAL,
    ConversationAction.CHANGE_TOPIC,
}


# ---- State 定义 ----

class CogniaState(TypedDict, total=False):
    """图状态（total=False 使所有字段可缺省）。

    序列化约束：所有字段必须是 JSON 原生类型（dict/str/list/int/bool），严禁直接存
    Pydantic 对象或自定义 Enum——msgpack checkpoint 无法安全反序列化。Pydantic 对象
    统一 `.model_dump(mode="json")` 转 dict 存、读取时 `.model_validate()` 还原。

    `messages` 与 `proficiency_deltas` 用 operator.add 累积（追加不覆盖）。
    """

    messages: Annotated[list, operator.add]          # 对话历史 [{"role","content"}]
    user_message: str                                 # 本轮用户原始输入
    goal: str
    goal_feedback: str
    knowledge_model: dict                             # KnowledgeModel.model_dump(mode="json")
    point_index: int
    current_point_id: str
    pending_question: str                             # 最近一次探针问题
    user_answer: str                                  # 用户对探针的回答
    diagnosis: dict                                   # Diagnosis.model_dump(mode="json")
    verification: dict                                # VerificationState.model_dump(mode="json")
    last_intervention: str
    intervention_fail_count: int
    probe_stall_count: int
    loop_count: int
    current_long_state: str | None                    # CognitiveState.value
    state_before_intervention: str | None             # CognitiveState.value
    proficiency_deltas: Annotated[list, operator.add]  # list[ProficiencyEntry.model_dump(mode="json")]
    ended: bool
    # Agent 编排字段
    agent_intent: str
    agent_action: str
    agent_reply: str
    proposed_goal: str | None
    target_point_id: str | None
    turn_phase: str                                   # "interpret" | "reply"
    engine_feedback: str                              # 引擎给 Agent 的结果摘要
    reasoning_trace: Annotated[list, operator.add]    # 每轮对话 Agent 的思考链片段（基座能力）


# ---- 路由函数（纯逻辑）----

def route_after_agent(state: CogniaState) -> str:
    """Agent 之后的路由：需要引擎的动作进引擎，纯对话动作等待用户，结束则 END。

    reply 阶段（引擎已处理完）无论动作如何都进入 await_user（或 ended 时 END），
    避免 Agent 在 reply 阶段再次触发引擎动作造成无限循环。
    """
    action = state.get("agent_action")
    if state.get("turn_phase") == "reply":
        return END if state.get("ended") else "await_user"
    if action == ConversationAction.END.value:
        return END
    if action in {a.value for a in ENGINE_ACTIONS}:
        return "learning_engine"
    return "await_user"


# ---- 组装图 ----

def build_graph(
    planner_model=None,
    teacher_model=None,
    diagnoser_model=None,
    checkpointer=None,
    store=None,
):
    """构建并编译 Cognia 对话 Agent ⇄ 学习引擎循环图。

    - planner_model：知识模型构建（学习引擎 SET_GOAL 使用）
    - teacher_model：对话 Agent 模型（理解意图 + 决策 + 生成回复）
    - diagnoser_model：诊断 + 双重验证（学习引擎 DIAGNOSE 使用）
    - checkpointer / store：持久化（生产注入 Postgres，测试注入内存桩）

    节点以闭包方式捕获各自模型与 store，避免模块级全局状态在多个图实例（测试 /
    多会话）间串扰。
    """
    planner = planner_model if planner_model is not None else models.get_planner_model()
    agent_model = teacher_model if teacher_model is not None else models.get_conversation_agent_model()
    diagnoser = diagnoser_model if diagnoser_model is not None else models.get_diagnoser_model()

    def conversation_agent(state: CogniaState, config: RunnableConfig) -> dict:
        """对话 Agent 节点：理解用户 + 决定动作 + 生成回复（无认知裁决权）。"""
        turn_phase = state.get("turn_phase", "interpret")
        user_id = get_user_id(config)
        profile = get_profile_dict(store, user_id) if (store is not None and user_id) else {}
        turn, reasoning = models.structured_output_with_reasoning(
            agent_model, ConversationTurn, build_agent_messages(state, profile)
        )

        updates: dict = {
            "agent_intent": turn.intent.value,
            "agent_action": turn.action.value,
            "agent_reply": turn.reply,
            "proposed_goal": turn.proposed_goal,
            "target_point_id": turn.target_point_id,
        }

        if reasoning:
            updates["reasoning_trace"] = [reasoning]

        if turn.action == ConversationAction.PROBE:
            updates["pending_question"] = turn.reply
        if turn.action == ConversationAction.EXPLAIN:
            updates["last_intervention"] = turn.reply
        if turn.action == ConversationAction.END:
            updates["ended"] = True

        # 对话历史：interpret 阶段追加用户输入；agent 有回复时追加 assistant 回复。
        new_msgs: list = []
        if turn_phase == "interpret":
            user_msg = state.get("user_message") or state.get("user_answer") or ""
            if user_msg:
                new_msgs.append({"role": "user", "content": user_msg})
        if turn.reply:
            new_msgs.append({"role": "assistant", "content": turn.reply})
        if new_msgs:
            updates["messages"] = new_msgs

        return updates

    def _handle_set_goal(state: CogniaState, config: RunnableConfig) -> dict:
        """构建 / 重建知识模型并初始化学习状态（跨会话读回历史熟练度）。

        load-or-build（Task ⑨）：同一 (user_id, 归一化 goal) 复用已冻结的
        知识模型，保证 point_id 跨会话稳定。否则 LLM 每次现生成 point_id，
        同一目标两次拆解得到不同 id，按 point_id 精确读回历史熟练度会查空，
        跨会话读回闭环无法真正闭合。
        """
        goal = (state.get("proposed_goal") or state.get("goal")
                or state.get("user_message") or "").strip()
        user_id = get_user_id(config)
        goal_key = normalize_goal(goal)

        km_dict = None
        if store is not None and user_id and goal_key:
            km_dict = get_knowledge_model(store, user_id, goal_key)

        if km_dict is not None:
            km = _load_knowledge_model(km_dict)
            built = False
        else:
            km = build_knowledge_model(planner, goal)
            km.goal = goal
            # 首次构建后冻结持久化：后续同目标直接复用，不再重生成
            if store is not None and user_id and goal_key:
                put_knowledge_model(store, user_id, goal_key, _dump(km))
            built = True

        current_point_id = km.points[0].id if km.points else None

        updates: dict = {
            "goal": goal,
            "knowledge_model": _dump(km),
            "point_index": 0,
            "current_point_id": current_point_id,
            "verification": _dump(VerificationState()),
            "intervention_fail_count": 0,
            "probe_stall_count": 0,
            "loop_count": 0,
            "state_before_intervention": None,
            "ended": False,
        }

        if current_point_id is not None:
            updates["current_long_state"] = read_historical_state(store, user_id, current_point_id)
        else:
            updates["current_long_state"] = None

        verb = "构建" if built else "复用"
        updates["engine_feedback"] = (
            f"已{verb}「{goal}」的知识模型，共 {len(km.points)} 个知识点。"
        )
        updates["turn_phase"] = "reply"
        return updates

    def _handle_diagnose(state: CogniaState, config: RunnableConfig) -> dict:
        """诊断 → 验证 → 状态机迁移裁决 → 选下一步 / 结束（学习事实层）。

        任何 Delta 都经 resolve_migration（高置信度 + mastered 双重验证 + 拓扑合法）
        三层闸门，Agent 无法旁路。
        """
        km = _load_knowledge_model(state.get("knowledge_model"))
        idx = state.get("point_index", 0)
        point = km.points[idx]
        user_answer = state.get("user_answer") or state.get("user_message") or ""
        question = state.get("pending_question", "")

        diagnosis = run_diagnosis(diagnoser, point, question, user_answer)

        updates: dict = {
            "diagnosis": _dump(diagnosis),
            "loop_count": state.get("loop_count", 0) + 1,
            "probe_stall_count": (
                0 if diagnosis.confidence == Confidence.HIGH
                else state.get("probe_stall_count", 0) + 1
            ),
        }

        verification = None
        genuinely_mastered = False
        fail_increment = 0

        before = _load_state(state.get("state_before_intervention"))
        if before is not None and not improved(diagnosis, before):
            fail_increment = 1

        if diagnosis.confidence == Confidence.HIGH:
            if diagnosis.state == CognitiveState.MASTERED:
                verification = run_verification(diagnoser, point, user_answer, question, diagnosis)
                updates["verification"] = _dump(verification)
                if is_mastered_migration_allowed(verification):
                    genuinely_mastered = True
                    apply_migration(
                        updates, diagnosis,
                        _load_state(state.get("current_long_state")), verification,
                    )
                else:
                    fail_increment += 1
                    diagnosis = diagnosis.model_copy(update={"state": CognitiveState.PARTIAL})
                    updates["diagnosis"] = _dump(diagnosis)
                    updates["state_before_intervention"] = state.get("current_long_state")
            else:
                apply_migration(
                    updates, diagnosis,
                    _load_state(state.get("current_long_state")),
                    _load_verification(state.get("verification")),
                )
                updates["state_before_intervention"] = (
                    updates.get("current_long_state") or state.get("current_long_state")
                )

        if fail_increment:
            updates["intervention_fail_count"] = state.get("intervention_fail_count", 0) + fail_increment

        fail_count = updates.get("intervention_fail_count", state.get("intervention_fail_count", 0))
        stall_count = updates.get("probe_stall_count", state.get("probe_stall_count", 0))

        if genuinely_mastered:
            if idx + 1 < len(km.points):
                nxt = km.points[idx + 1]
                updates["point_index"] = idx + 1
                updates["current_point_id"] = nxt.id
                updates["verification"] = _dump(VerificationState())
                updates["intervention_fail_count"] = 0
                updates["probe_stall_count"] = 0
                updates["state_before_intervention"] = None
                updates["current_long_state"] = read_historical_state(
                    store, get_user_id(config), nxt.id
                )
            else:
                updates["ended"] = True
                updates["goal_feedback"] = "🎉 本次学习目标已达成！"
        elif fail_count >= MAX_INTERVENTION_FAILS:
            updates["ended"] = True
            updates["goal_feedback"] = (
                "这个知识点我们暂时先告一段落。你可以稍后再试，"
                "或换一个更具体、更聚焦的学习目标继续。"
            )
        elif stall_count >= MAX_PROBE_STALLS:
            updates["ended"] = True
            updates["goal_feedback"] = (
                "这个知识点我们暂时先告一段落。你可以稍后再试，"
                "或换一个更具体、更聚焦的学习目标继续。"
            )

        parts = [f"诊断：{diagnosis.state.value}（置信度 {diagnosis.confidence.value}）"]
        if verification is not None:
            parts.append(f"双重验证：概念 {verification.concept.value} / 场景 {verification.scenario.value}")
        if genuinely_mastered:
            parts.append("已通过双重验证，正式掌握该知识点")
        elif diagnosis.confidence == Confidence.HIGH and diagnosis.state != CognitiveState.MASTERED:
            parts.append("已迁移到该状态，但仍需教学")
        elif diagnosis.confidence in (Confidence.MEDIUM, Confidence.LOW):
            parts.append(f"证据不足/存在歧义（连续 {stall_count} 轮无进展），需追问或换方式讲解")
        if updates.get("ended"):
            parts.append("学习已结束" if genuinely_mastered else "已放弃该知识点")
        updates["engine_feedback"] = "；".join(parts)

        updates["turn_phase"] = "reply"
        return updates

    def learning_engine(state: CogniaState, config: RunnableConfig) -> dict:
        """学习引擎节点：执行学习事实的合法变更（无对话自主权）。"""
        action = state.get("agent_action")
        if action in (ConversationAction.SET_GOAL.value, ConversationAction.CHANGE_TOPIC.value):
            return _handle_set_goal(state, config)
        return _handle_diagnose(state, config)

    def await_user(state: CogniaState) -> dict:
        """interrupt 暂停：把 Agent 的回复交给用户，等待下一轮输入。"""
        answer = interrupt({"type": "conversation", "message": state.get("agent_reply", "")})
        return {
            "user_answer": answer,
            "user_message": answer,
            "turn_phase": "interpret",
            "engine_feedback": None,
        }

    builder = StateGraph(CogniaState)
    builder.add_node("conversation_agent", conversation_agent)
    builder.add_node("learning_engine", learning_engine)
    builder.add_node("await_user", await_user)

    builder.add_edge(START, "conversation_agent")
    builder.add_conditional_edges(
        "conversation_agent",
        route_after_agent,
        {"learning_engine": "learning_engine", "await_user": "await_user", END: END},
    )
    builder.add_edge("learning_engine", "conversation_agent")
    builder.add_edge("await_user", "conversation_agent")

    return builder.compile(checkpointer=checkpointer)
