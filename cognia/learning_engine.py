"""Learning Engine —— Cognia 的学习事实层。

职责（与 Conversation Agent 解耦，本次架构重构核心原则之一）：
- 承载并合法变更学习状态：当前目标、当前知识点、认知状态、证据、诊断、验证、熟练度。
- 诊断（Diagnosis）只产出「候选」，状态迁移必须经 state_machine 裁决
  （resolve_migration 三层闸门：高置信度 + mastered 双重验证 + 拓扑合法）。
- 掌握判定必须「概念解释 + 场景辨析」双重验证，禁止单次问答直写 mastered。

本模块**不负责「怎么和用户交流」**（那是对话 Agent 的职责），只负责
「什么状态变化是合法的」以及承载这些状态的读写。
"""

from pydantic import BaseModel, Field

from cognia import models
from cognia.memory import get_current_proficiency
from cognia.schemas import (
    CognitiveState,
    Confidence,
    Diagnosis,
    KnowledgeModel,
    KnowledgePoint,
    ProficiencyEntry,
    ValidationResult,
    VerificationState,
)
from cognia.state_machine import can_transition, is_mastered_migration_allowed
from cognia.prompts.learning_engine import (
    CONCEPT_VERIFIER_SYSTEM_PROMPT,
    DIAGNOSER_SYSTEM_PROMPT,
    KNOWLEDGE_MODELER_SYSTEM_PROMPT,
    SCENARIO_VERIFIER_SYSTEM_PROMPT,
)


# ---- 纯函数：迁移裁决（诊断 ≠ 迁移 的核心）----

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
    if diagnosis.confidence != Confidence.HIGH:
        return None

    to_state = diagnosis.state

    if to_state == CognitiveState.MASTERED:
        if verification is None or not is_mastered_migration_allowed(verification):
            return None

    effective_from = from_state if from_state is not None else CognitiveState.UNASSESSED
    if not can_transition(effective_from, to_state):
        return None

    return ProficiencyEntry(
        point_id=diagnosis.point_id,
        from_state=from_state,
        to_state=to_state,
        evidence=diagnosis.evidence,
    )


def apply_migration(
    updates: dict,
    diagnosis: Diagnosis,
    current_long_state: CognitiveState | None,
    verification: VerificationState | None,
) -> None:
    """尝试产生 Proficiency Delta 并写入 updates。

    任何 Delta 都经 resolve_migration 三层闸门，返回 None 时不修改 updates（不迁移）。
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


def improved(diagnosis: Diagnosis, before: CognitiveState) -> bool:
    """判断诊断状态相对干预前是否改善。

    教学启发式排序：unknown < misconception < partial < mastered。
    """
    if diagnosis.state == CognitiveState.MASTERED:
        return True
    if before == CognitiveState.MASTERED:
        return False  # 从 mastered 降级 = 未改善
    new_rank = _STATE_PROGRESS.get(diagnosis.state, -1)
    old_rank = _STATE_PROGRESS.get(before, -1)
    return new_rank > old_rank


# ---- 序列化 / 还原 helper ----

def _dump(model) -> dict:
    """Pydantic 对象 → JSON 可序列化 dict（msgpack checkpoint 安全）。"""
    return model.model_dump(mode="json")


def _load_state(value: str | None) -> CognitiveState | None:
    return CognitiveState(value) if value is not None else None


def _load_diagnosis(value: dict | None) -> Diagnosis | None:
    return Diagnosis.model_validate(value) if value is not None else None


def _load_verification(value: dict | None) -> VerificationState | None:
    return VerificationState.model_validate(value) if value is not None else None


def _load_knowledge_model(value: dict | None) -> KnowledgeModel | None:
    return KnowledgeModel.model_validate(value) if value is not None else None


# ---- 知识模型构建 ----

def build_knowledge_model(planner, goal: str) -> KnowledgeModel:
    """构建知识模型：普通文本调用 + 扁平 JSON 数组解析。

    背景：adapter 上游工蜂 Gateway 对带嵌套对象引用（$defs/$ref）的 JSON Schema 的
    function calling 支持有缺陷，KnowledgeModel 改用「普通 invoke + 输出扁平 JSON
    数组 + 手动解析」。

    兼容测试桩：ScriptedLLM 的 invoke 直接返回 KnowledgeModel 对象（而非字符串），
    isinstance 命中后原样返回。
    """
    messages = [
        ("system", KNOWLEDGE_MODELER_SYSTEM_PROMPT),
        ("human", (
            f"学习目标：{goal}\n\n"
            "请只输出一个 JSON 数组，每个元素是一个知识点对象，格式如下"
            "（不要输出任何解释、注释或 markdown 代码块）：\n"
            '[{"id": "唯一标识", "name": "知识点名称", "description": "一句话描述", '
            '"prerequisites": ["依赖的知识点id"]}]'
        )),
    ]
    result = planner.invoke(messages)
    if isinstance(result, KnowledgeModel):
        return result
    text = result.content if hasattr(result, "content") else str(result)
    try:
        data = models._extract_json(text)
    except ValueError as exc:
        # 解析失败（截断 / 语法错误等）：把错误回喂给模型，让其修正后重试一次，
        # 避免一次坏 JSON 直接让 LangGraph 工具节点抛错、终止整个 agent 运行。
        retry_result = planner.invoke([
            *messages,
            ("human", (
                f"你上一次的输出无法解析为合法 JSON，错误信息：{exc}\n"
                "请重新只输出一个合法 JSON 数组，不要输出任何解释、注释或 markdown 代码块。"
            )),
        ])
        if isinstance(retry_result, KnowledgeModel):
            return retry_result
        retry_text = retry_result.content if hasattr(retry_result, "content") else str(retry_result)
        data = models._extract_json(retry_text)
    points = [KnowledgePoint.model_validate(p) for p in data]
    return KnowledgeModel(goal=goal, points=points)


# ---- 诊断与验证（LLM 判定，候选而非裁决）----

class _ConceptAssessment(BaseModel):
    """verify 的概念解释验证结果。"""

    passed: bool = Field(description="概念解释是否准确、完整")
    evidence: str = Field(default="", description="用户表达中支撑判定的原话片段")


class _ScenarioAssessment(BaseModel):
    """verify 的场景/边界辨析验证结果。"""

    passed: bool = Field(description="场景/边界/反例辨析是否通过")
    evidence: str = Field(default="", description="用户表达中支撑判定的原话片段")


def run_diagnosis(diagnoser, point: KnowledgePoint, question: str, user_answer: str) -> Diagnosis:
    """基于用户表达诊断五态 + 置信度 + 证据（独立严格 prompt）。

    产出的是「候选」，最终迁移由 resolve_migration + state_machine 裁决。
    """
    diagnosis = models.structured_output(diagnoser, Diagnosis, [
        ("system", DIAGNOSER_SYSTEM_PROMPT),
        ("human", (
            f"知识点：{point.name}（{point.description}）\n"
            f"探针问题：{question}\n"
            f"用户回答：{user_answer}\n\n请诊断。"
        )),
    ])
    diagnosis.point_id = point.id
    return diagnosis


def run_verification(
    diagnoser,
    point: KnowledgePoint,
    user_answer: str,
    question: str,
    diagnosis: Diagnosis,
) -> VerificationState:
    """双重验证：概念解释 + 场景辨析，均独立 LLM 判定（不白送）。

    验证失败时（任一未过）由调用方将 diagnosis.state 降级为 partial 并计数。
    """
    concept_assessment = models.structured_output(diagnoser, _ConceptAssessment, [
        ("system", CONCEPT_VERIFIER_SYSTEM_PROMPT),
        ("human", f"知识点：{point.name}（{point.description}）\n探针问题：{question}\n用户回答：{user_answer}\n\n请判断概念解释是否通过。"),
    ])
    scenario_assessment = models.structured_output(diagnoser, _ScenarioAssessment, [
        ("system", SCENARIO_VERIFIER_SYSTEM_PROMPT),
        ("human", f"知识点：{point.name}（{point.description}）\n用户回答：{user_answer}\n\n请判断场景辨析是否通过。"),
    ])

    concept_result = ValidationResult.PASSED if concept_assessment.passed else ValidationResult.FAILED
    scenario_result = ValidationResult.PASSED if scenario_assessment.passed else ValidationResult.FAILED

    # concept 通过时用评估证据（缺省才回填 diagnosis 的 mastered 证据）；
    # concept 失败时不得回填 diagnosis.evidence（那是「判 mastered」的正面证据）。
    if concept_assessment.evidence:
        concept_evidence = [concept_assessment.evidence]
    elif concept_result == ValidationResult.PASSED:
        concept_evidence = diagnosis.evidence
    else:
        concept_evidence = []

    return VerificationState(
        concept=concept_result,
        scenario=scenario_result,
        concept_evidence=concept_evidence,
        scenario_evidence=[scenario_assessment.evidence] if scenario_assessment.evidence else [],
        current_step="done",
    )


def read_historical_state(store, user_id: str, point_id: str) -> str | None:
    """跨会话读回某知识点的历史熟练度（store 为 None 时返回 None=unassessed）。"""
    if store is None or not user_id or not point_id:
        return None
    return get_current_proficiency(store, user_id, point_id)
