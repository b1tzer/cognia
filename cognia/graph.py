"""Cognia 教学核心图（LangGraph StateGraph）。

实现 plan §3 的完整状态机：单 Agent + 单 Stateful Graph 多节点。

节点（6 个逻辑节点 + 1 个 interrupt 专用节点 + 1 个验证节点 = 8 个注册节点）：
- setup_goal：接收目标，过大则引导缩小（LLM）
- build_model：生成知识模型（LLM）
- probe：生成探针问题（LLM）
- await_answer：interrupt 暂停，等待用户表达（无 LLM，专用无副作用节点）
- diagnose：基于表达诊断五态 + 置信度 + 证据（LLM，独立严格 prompt）
- verify：双重验证（概念 + 场景辨析，LLM）
- intervene：生成干预动作（LLM）
- select_next：掌握后选下一个知识点或结束（纯逻辑）

关键设计（与 state-machine.md 严格对齐）：
- 「诊断是候选，迁移是裁决」：`resolve_migration` 是三段闸门的唯一裁决入口，
  任何 Proficiency Delta 只能由此函数产生（state-machine §5 规则 1）。
- interrupt 放在 `await_answer` 专用节点（开头即 interrupt），避免重跑 LLM。
- 干预失败判定：一次「干预→探测→表达→诊断」闭环后状态未改善才计数。
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from cognia import models
from cognia.memory import get_current_proficiency
from cognia.schemas import (
    CognitiveState,
    Confidence,
    Diagnosis,
    KnowledgeModel,
    ProficiencyEntry,
    ValidationResult,
    VerificationState,
)
from cognia.state_machine import (
    can_transition,
    is_mastered_migration_allowed,
)

# ---- 常量 ----

MAX_INTERVENTION_FAILS = 3  # 单知识点失败轮次上限（含干预失败与 verify 失败），达此值触发回溯（spec §6）

DIAGNOSER_SYSTEM_PROMPT = """你是 Cognia 的认知诊断器。你的唯一职责是：基于用户对当前知识点的表达，判定其认知状态（五态之一）与置信度（三级之一），并给出支撑判定的用户原话证据。

## 五态定义（spec v2.0）
- unassessed（未评估）：无法区分用户是否掌握，或未经探测，证据不足时一律保持此态。
- unknown（盲区）：用户明确表示不知道、没听过、没接触过（必须有明确否定证据）。
- partial（部分掌握）：能说大意但细节模糊，边界说不清。
- misconception（错误理解）：表达有明显逻辑错误、概念混淆、因果倒置，且用户自认为正确。
- mastered（已掌握）：概念解释正确 + 场景/边界辨析正确（两份独立正向证据）。

## 置信度三级（clarifications Q4，行为分级，非百分比）
- high（高置信度）：证据充分，可直接判定。
- medium（中置信度）：存在歧义，状态应原地冻结。
- low（低置信度）：证据不足，不改变状态。

## 铁律
1. 证据必须来自用户原话片段，严禁脑补（spec §6）。
2. 「不知道」≠「答不好」：unknown 必须有明确否定证据；有明显逻辑错误应判 misconception/partial，不得粗暴判 unknown。
3. 无法区分时一律 unassessed。
4. 你只输出诊断候选，不负责最终状态迁移（迁移由状态机裁决）。"""


# ---- 内部辅助 Pydantic 模型（仅图内部使用，不入 schemas.py）----

class _GoalAssessment(BaseModel):
    """setup_goal 的目标合理性判断。"""

    too_broad: bool = Field(description="目标是否过大/过宽泛，需要缩小")
    feedback: str = Field(default="", description="若过大，引导用户缩小目标的建议")


class _ScenarioAssessment(BaseModel):
    """verify 的场景/边界辨析验证结果。"""

    passed: bool = Field(description="场景/边界/反例辨析是否通过")
    evidence: str = Field(default="", description="用户表达中支撑判定的原话片段")


class _ConceptAssessment(BaseModel):
    """verify 的概念解释验证结果。"""

    passed: bool = Field(description="概念解释是否准确、完整")
    evidence: str = Field(default="", description="用户表达中支撑判定的原话片段")


# ---- State 定义 ----

class CogniaState(TypedDict, total=False):
    """图状态（total=False 使所有字段可缺省）。

    序列化约束（任务⑥前置）：所有字段必须是 JSON 原生类型（dict/str/list/int/bool），
    严禁直接存 Pydantic 对象或自定义 Enum——msgpack checkpoint 无法安全反序列化，
    当前仅告警（`Deserializing unregistered type ... will be blocked in a future version`）。
    Pydantic 对象统一 `.model_dump(mode="json")` 转 dict 存、读取时 `.model_validate()` 还原；
    CognitiveState 统一存 `.value`（str）。

    注意：`messages` 与 `proficiency_deltas` 用 operator.add 累积（追加不覆盖）。
    `current_long_state` 为 None 表示「未评估（unassessed）」。
    """

    messages: Annotated[list, operator.add]
    goal: str
    goal_feedback: str
    knowledge_model: dict  # KnowledgeModel.model_dump(mode="json")
    point_index: int
    current_point_id: str
    pending_question: str
    user_answer: str
    diagnosis: dict  # Diagnosis.model_dump(mode="json")
    verification: dict  # VerificationState.model_dump(mode="json")
    last_intervention: str
    intervention_fail_count: int
    loop_count: int
    current_long_state: str | None  # CognitiveState.value
    state_before_intervention: str | None  # CognitiveState.value
    proficiency_deltas: Annotated[list, operator.add]  # list[ProficiencyEntry.model_dump(mode="json")]
    ended: bool


# ---- 序列化边界 helper（Pydantic <-> JSON dict，Enum <-> str）----

def _dump(model) -> dict:
    """Pydantic 对象 → JSON 可序列化 dict。

    mode="json" 会把嵌套的 Enum 转成 .value（str）、datetime 转成 ISO 字符串，
    确保产物是纯 JSON 原生类型，可被 msgpack checkpoint 安全序列化。
    """
    return model.model_dump(mode="json")


def _load_state(value: str | None) -> CognitiveState | None:
    """str → CognitiveState（None 透传）。"""
    return CognitiveState(value) if value is not None else None


def _load_diagnosis(value: dict | None) -> Diagnosis | None:
    return Diagnosis.model_validate(value) if value is not None else None


def _load_verification(value: dict | None) -> VerificationState | None:
    return VerificationState.model_validate(value) if value is not None else None


def _load_knowledge_model(value: dict | None) -> KnowledgeModel | None:
    return KnowledgeModel.model_validate(value) if value is not None else None


# ---- 纯函数：迁移裁决（诊断≠迁移 的核心）----

def resolve_migration(
    diagnosis: Diagnosis,
    from_state: CognitiveState | None,
    verification: VerificationState | None,
) -> ProficiencyEntry | None:
    """裁决一次诊断候选是否产生状态迁移（产生 Proficiency Delta）。

    这是「诊断是候选、迁移是裁决」的唯一入口（state-machine §5 规则 1）。
    三层闸门缺一不可：

    1. 置信度 == HIGH（高置信度是唯一迁移门槛）
    2. 若目标为 mastered，双重验证必须全过
    3. 拓扑合法（can_transition）

    返回 None 表示不迁移（不产生任何 Delta）；返回 ProficiencyEntry 表示迁移。
    `from_state` 为 None 表示首次诊断（等价 unassessed）。
    """
    # 闸门 3：高置信度
    if diagnosis.confidence != Confidence.HIGH:
        return None

    to_state = diagnosis.state

    # 闸门 2：mastered 双重验证
    if to_state == CognitiveState.MASTERED:
        if verification is None or not is_mastered_migration_allowed(verification):
            return None

    # 闸门 1：拓扑合法
    effective_from = from_state if from_state is not None else CognitiveState.UNASSESSED
    if not can_transition(effective_from, to_state):
        return None

    return ProficiencyEntry(
        point_id=diagnosis.point_id,
        from_state=from_state,
        to_state=to_state,
        evidence=diagnosis.evidence,
    )


def _apply_migration(
    updates: dict,
    diagnosis: Diagnosis,
    current_long_state: CognitiveState | None,
    verification: VerificationState | None,
) -> None:
    """尝试产生 Proficiency Delta 并写入 updates。

    这是「诊断是候选、迁移是裁决」的唯一落地处：任何 Delta 都经由
    resolve_migration 三层闸门，返回 None 时不修改 updates（不迁移）。
    diagnose 与 select_next 两节点共用此函数，避免「写 delta + 更新状态」
    两处重复、将来改一处漏一处。
    """
    entry = resolve_migration(diagnosis, current_long_state, verification)
    if entry is not None:
        updates["proficiency_deltas"] = [_dump(entry)]
        updates["current_long_state"] = entry.to_state.value


# ---- 纯函数：干预失败判定 ----

_STATE_PROGRESS = {
    CognitiveState.UNKNOWN: 0,
    CognitiveState.MISCONCEPTION: 1,
    CognitiveState.PARTIAL: 2,
    CognitiveState.MASTERED: 3,
}


def _improved(diagnosis: Diagnosis, before: CognitiveState) -> bool:
    """判断诊断状态相对干预前是否改善。

    「改善」的教学启发式排序：unknown < misconception < partial < mastered。
    调用方保证 before 是具体状态（进入 intervene 前已迁移成功，
    current_long_state 永不为 None/unassessed）。
    """
    if diagnosis.state == CognitiveState.MASTERED:
        return True
    if before == CognitiveState.MASTERED:
        return False  # 从 mastered 降级 = 未改善
    new_rank = _STATE_PROGRESS.get(diagnosis.state, -1)
    old_rank = _STATE_PROGRESS.get(before, -1)
    return new_rank > old_rank


# ---- 路由函数（纯逻辑，模块级）----

def route_after_setup_goal(state: CogniaState) -> str:
    """setup_goal 之后：目标过大则结束，否则进入建模型。"""
    return END if state.get("ended") else "build_model"


def route_after_diagnose(state: CogniaState) -> str:
    """diagnose 之后按置信度 + 失败计数路由。"""
    diagnosis = _load_diagnosis(state.get("diagnosis"))
    if diagnosis is None:
        return "probe"

    # 防失控：干预失败达上限 → 回溯（放弃当前知识点）
    if state.get("intervention_fail_count", 0) >= MAX_INTERVENTION_FAILS:
        return "select_next"

    if diagnosis.confidence == Confidence.HIGH:
        if diagnosis.state == CognitiveState.MASTERED:
            return "verify"  # 高置信度 mastered → 双重验证
        return "intervene"   # 高置信度非 mastered → 干预
    # 中/低置信度：不改变状态，重新探测
    return "probe"


def route_after_verify(state: CogniaState) -> str:
    """verify 之后：验证通过 → select_next；不通过 → 重新干预。"""
    verification = _load_verification(state.get("verification"))
    if verification is not None and is_mastered_migration_allowed(verification):
        return "select_next"
    return "intervene"


def route_after_select_next(state: CogniaState) -> str:
    """select_next 之后：有下一个知识点则继续探测，否则结束。"""
    return END if state.get("ended") else "probe"


# ---- 节点实现 ----

def _extract_text(result) -> str:
    """从 LLM 返回结果中提取文本（兼容 str / AIMessage）。"""
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        return result.content
    return str(result)


def build_graph(
    planner_model=None,
    teacher_model=None,
    diagnoser_model=None,
    checkpointer=None,
    store=None,
):
    """构建并编译 Cognia 教学核心图。

    模型、checkpointer、store 均可注入（默认用 models.py 工厂），便于测试用 fake 替换。
    生产环境必须注入持久化 checkpointer（PostgresSaver），否则 interrupt 无法恢复。
    store（BaseStore）用于跨会话读回历史熟练度（Q6：读→推理→写闭环）；
    为 None 时按「首次评估」处理（current_long_state 保持 None/unassessed）。
    """
    planner = planner_model if planner_model is not None else models.get_planner_model()
    teacher = teacher_model if teacher_model is not None else models.get_teacher_model()
    diagnoser = diagnoser_model if diagnoser_model is not None else models.get_diagnoser_model()

    def setup_goal(state: CogniaState) -> dict:
        """接收目标，判断是否过大；过大则引导缩小。"""
        goal = state["goal"]
        assessment = planner.with_structured_output(_GoalAssessment).invoke([
            ("system", "你是 Cognia 的目标引导器。判断学习目标是否过大/过宽泛（主题过多、范围过广、无法在单次会话聚焦）。"),
            ("human", f"学习目标：{goal}"),
        ])
        updates: dict = {
            "loop_count": 0,
            "intervention_fail_count": 0,
            "point_index": 0,
            "verification": _dump(VerificationState()),
            "current_long_state": None,
            "state_before_intervention": None,
        }
        if assessment.too_broad:
            updates["ended"] = True
            updates["goal_feedback"] = assessment.feedback
        return updates

    def build_model(state: CogniaState, config: RunnableConfig) -> dict:
        """生成知识模型，聚焦第一个知识点，并跨会话读回其历史熟练度。"""
        goal = state["goal"]
        km = planner.with_structured_output(KnowledgeModel).invoke([
            ("system", "你是 Cognia 的知识建模器。将学习目标拆解为 5~15 个知识点及其依赖关系。"),
            ("human", f"学习目标：{goal}"),
        ])
        # 强制目标一致，并取第一个知识点
        km.goal = goal
        current_point_id = km.points[0].id if km.points else None
        updates: dict = {
            "knowledge_model": _dump(km),
            "current_point_id": current_point_id,
            "point_index": 0,
        }
        # 读侧（Q6）：跨会话恢复第一个知识点的历史熟练度，初始化 current_long_state，
        # 使教学/诊断从历史态继续，而非永远从 unassessed 重新开始。
        # store 为 None（如部分测试）时跳过，保持「首次评估」语义。
        if store is not None and current_point_id is not None:
            user_id = config.get("configurable", {}).get("user_id")
            if user_id:
                historical = get_current_proficiency(store, user_id, current_point_id)
                if historical is not None:
                    updates["current_long_state"] = historical
        return updates

    def probe(state: CogniaState) -> dict:
        """生成探针问题（针对当前知识点）。"""
        km = _load_knowledge_model(state["knowledge_model"])
        idx = state.get("point_index", 0)
        point = km.points[idx]
        last_intervention = state.get("last_intervention")
        context = f"\n上一轮干预：{last_intervention}" if last_intervention else ""
        result = teacher.invoke([
            ("system", "你是 Cognia 教学教练。生成一个探针问题，探测用户对当前知识点的真实认知状态（区分 partial / misconception / unknown / mastered）。"),
            ("human", f"知识点：{point.name}（{point.description}）{context}\n请生成探针问题。"),
        ])
        return {"pending_question": _extract_text(result), "user_answer": None}

    def await_answer(state: CogniaState) -> dict:
        """interrupt 暂停，等待用户表达（开头即 interrupt，无副作用）。"""
        answer = interrupt({
            "type": "probe",
            "question": state["pending_question"],
        })
        return {"user_answer": answer}

    def diagnose(state: CogniaState) -> dict:
        """基于用户表达诊断五态 + 置信度 + 证据（独立严格 prompt）。"""
        km = _load_knowledge_model(state["knowledge_model"])
        idx = state.get("point_index", 0)
        point = km.points[idx]
        user_answer = state.get("user_answer", "")
        question = state.get("pending_question", "")

        diagnosis = diagnoser.with_structured_output(Diagnosis).invoke([
            ("system", DIAGNOSER_SYSTEM_PROMPT),
            ("human", f"知识点：{point.name}（{point.description}）\n探针问题：{question}\n用户回答：{user_answer}\n\n请诊断。"),
        ])
        # 强制绑定到当前知识点，防止 LLM 填错 point_id
        diagnosis.point_id = point.id

        updates: dict = {
            "diagnosis": _dump(diagnosis),
            "loop_count": state.get("loop_count", 0) + 1,
        }

        # 非 mastered 高置信度：立即尝试迁移（诊断≠迁移的唯一落地处）
        if diagnosis.confidence == Confidence.HIGH and diagnosis.state != CognitiveState.MASTERED:
            _apply_migration(
                updates,
                diagnosis,
                _load_state(state.get("current_long_state")),
                _load_verification(state.get("verification")),
            )

        # 干预失败判定：仅当存在「干预前状态」时（即已经历过至少一轮干预）
        before = _load_state(state.get("state_before_intervention"))
        if before is not None:
            if not _improved(diagnosis, before):
                updates["intervention_fail_count"] = state.get("intervention_fail_count", 0) + 1

        return updates

    def verify(state: CogniaState) -> dict:
        """双重验证：概念解释 + 场景辨析，均独立 LLM 判定（不白送）。"""
        diagnosis = _load_diagnosis(state["diagnosis"])
        km = _load_knowledge_model(state["knowledge_model"])
        idx = state.get("point_index", 0)
        point = km.points[idx]
        user_answer = state.get("user_answer", "")
        question = state.get("pending_question", "")

        concept_assessment = diagnoser.with_structured_output(_ConceptAssessment).invoke([
            ("system", "你是 Cognia 的概念解释验证器。判断用户回答是否准确、完整地解释了当前知识点的核心概念。"),
            ("human", f"知识点：{point.name}（{point.description}）\n探针问题：{question}\n用户回答：{user_answer}\n\n请判断概念解释是否通过。"),
        ])
        scenario_assessment = diagnoser.with_structured_output(_ScenarioAssessment).invoke([
            ("system", "你是 Cognia 的场景辨析验证器。判断用户回答是否体现对场景/边界/反例的正确辨析能力。"),
            ("human", f"知识点：{point.name}（{point.description}）\n用户回答：{user_answer}\n\n请判断场景辨析是否通过。"),
        ])

        concept_result = ValidationResult.PASSED if concept_assessment.passed else ValidationResult.FAILED
        scenario_result = ValidationResult.PASSED if scenario_assessment.passed else ValidationResult.FAILED

        # concept 通过时用评估证据（缺省才回填 diagnosis 的 mastered 证据）；
        # concept 失败时不得回填 diagnosis.evidence（那是「判 mastered」的正面证据，语义矛盾）。
        if concept_assessment.evidence:
            concept_evidence = [concept_assessment.evidence]
        elif concept_result == ValidationResult.PASSED:
            concept_evidence = diagnosis.evidence
        else:
            concept_evidence = []

        verification = VerificationState(
            concept=concept_result,
            scenario=scenario_result,
            concept_evidence=concept_evidence,
            scenario_evidence=[scenario_assessment.evidence] if scenario_assessment.evidence else [],
            current_step="done",
        )

        updates: dict = {"verification": _dump(verification)}

        # 验证失败 = 未真正达到 mastered = 未改善，计数。
        # 否则「diagnose 判 mastered → verify 反复失败 → 再干预」会因
        # _improved(mastered, ...) 恒 True 而永不计数，形成无限循环。
        if not (concept_result == ValidationResult.PASSED and scenario_result == ValidationResult.PASSED):
            updates["intervention_fail_count"] = state.get("intervention_fail_count", 0) + 1
            # 同步降级 diagnosis.state：concept/scenario 任一失败都说明实际未达
            # mastered，降到 partial（「概念会场景不会 / 概念未说清」的典型特征）。
            # 避免 intervene 拿到「已 mastered + 请干预」的自相矛盾状态。
            updates["diagnosis"] = _dump(diagnosis.model_copy(update={"state": CognitiveState.PARTIAL}))

        return updates

    def intervene(state: CogniaState) -> dict:
        """生成干预动作，并记录干预前状态（用于下一轮失败判定）。"""
        km = _load_knowledge_model(state["knowledge_model"])
        idx = state.get("point_index", 0)
        point = km.points[idx]
        diagnosis = _load_diagnosis(state["diagnosis"])

        # 消费 verification 结果：verify 失败后路由到这里时，verification 携带
        # 「概念 vs 场景」哪块没过，据此生成针对性纠错（而非笼统「已 mastered 请干预」）。
        verification = _load_verification(state.get("verification"))
        gap_hint = ""
        if verification is not None:
            concept_failed = verification.concept == ValidationResult.FAILED
            scenario_failed = verification.scenario == ValidationResult.FAILED
            if concept_failed and scenario_failed:
                gap_hint = "\n验证缺口：概念解释与场景辨析均未通过，请针对核心概念本身做讲解/纠错。"
            elif concept_failed:
                gap_hint = "\n验证缺口：概念解释未通过，请针对核心概念做讲解/纠错。"
            elif scenario_failed:
                gap_hint = "\n验证缺口：场景/边界辨析未通过，请针对场景/边界/反例做辨析训练。"

        result = teacher.invoke([
            ("system", "你是 Cognia 教学教练。针对用户的认知问题生成干预动作（追问/解释/纠错）。"),
            ("human", f"知识点：{point.name}\n诊断状态：{diagnosis.state.value}（置信度 {diagnosis.confidence.value}）{gap_hint}\n请生成干预内容。"),
        ])
        return {
            "last_intervention": _extract_text(result),
            "state_before_intervention": state.get("current_long_state"),
        }

    def select_next(state: CogniaState) -> dict:
        """mastered 迁移裁决 + 选下一个知识点或结束（纯逻辑）。

        回溯语义：fail_count ≥ 3 到达本节点 = 放弃当前知识点、停止干预，
        **不回滚**已记录的历史 Delta（宪法 §5：增量 Delta，严禁全量重写）。
        """
        diagnosis = _load_diagnosis(state.get("diagnosis"))
        updates: dict = {}

        # mastered 迁移裁决（验证已在 route_after_verify 通过）
        if diagnosis is not None and diagnosis.confidence == Confidence.HIGH \
                and diagnosis.state == CognitiveState.MASTERED:
            _apply_migration(
                updates,
                diagnosis,
                _load_state(state.get("current_long_state")),
                _load_verification(state.get("verification")),
            )

        # 选下一个知识点
        km = _load_knowledge_model(state.get("knowledge_model"))
        idx = state.get("point_index", 0)
        if km is not None and idx + 1 < len(km.points):
            updates["point_index"] = idx + 1
            updates["current_point_id"] = km.points[idx + 1].id
            updates["verification"] = _dump(VerificationState())
            updates["intervention_fail_count"] = 0
            updates["state_before_intervention"] = None
            # 切到新知识点：重置长期状态为 None（新点从未评估）。
            # 否则会继承上一个点的 mastered，导致伪造「mastered→partial」降级 Delta，
            # 或 mastered 自我迁移被拒、新点永远无法 mastered。
            # 未来做「跨会话读回」时，此处应改为读回该点的历史态（而非硬置 None）。
            updates["current_long_state"] = None
        else:
            updates["ended"] = True
        return updates

    # ---- 组装图 ----

    builder = StateGraph(CogniaState)
    builder.add_node("setup_goal", setup_goal)
    builder.add_node("build_model", build_model)
    builder.add_node("probe", probe)
    builder.add_node("await_answer", await_answer)
    builder.add_node("diagnose", diagnose)
    builder.add_node("verify", verify)
    builder.add_node("intervene", intervene)
    builder.add_node("select_next", select_next)

    builder.add_edge(START, "setup_goal")
    builder.add_conditional_edges("setup_goal", route_after_setup_goal, {"build_model": "build_model", END: END})
    builder.add_edge("build_model", "probe")
    builder.add_edge("probe", "await_answer")
    builder.add_edge("await_answer", "diagnose")
    builder.add_conditional_edges(
        "diagnose",
        route_after_diagnose,
        {"verify": "verify", "intervene": "intervene", "probe": "probe", "select_next": "select_next"},
    )
    builder.add_conditional_edges("verify", route_after_verify, {"select_next": "select_next", "intervene": "intervene"})
    builder.add_edge("intervene", "probe")
    builder.add_conditional_edges("select_next", route_after_select_next, {"probe": "probe", END: END})

    return builder.compile(checkpointer=checkpointer)
