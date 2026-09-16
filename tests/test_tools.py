"""cognia.tools 认知模型工具层单元测试。

全部使用 ScriptedLLM 假模型 + InMemoryStore，离线、快速、不依赖 API。

重点验证：
1. 读工具：read_learner_state 正确读五态。
2. 写工具 propose_diagnosis 的三层闸门：
   - 中 / 低置信度不迁移；
   - mastered 必须概念 + 场景双重验证全过才迁移；
   - 验证失败不迁移；
   - 高置信度非 mastered 经状态机裁决迁移，并增量写回 store。
3. 教学工具 generate_probe / explain 生成文本。
4. build_learning_goal 的 load-or-build 复用。
"""

import json

from langgraph.store.memory import InMemoryStore

from cognia.schemas import (
    CognitiveState,
    Confidence,
    Diagnosis,
    KnowledgeModel,
    KnowledgePoint,
)
from cognia.tools import build_cognia_tools


class _Msg:
    """带 content 的假 AIMessage。"""

    def __init__(self, content):
        self.content = content


class ScriptedLLM:
    """按队列返回预设响应的假模型。

    兼容两种调用：
    - structured_output 路径：with_structured_output 返回 self，invoke 弹出对象。
    - 普通 invoke 路径：invoke 弹出 _Msg（带 content）。
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        assert self._responses, "ScriptedLLM 响应队列耗尽"
        return self._responses.pop(0)


class _ConceptAssessmentStub:
    def __init__(self, passed, evidence=""):
        self.passed = passed
        self.evidence = evidence


class _ScenarioAssessmentStub:
    def __init__(self, passed, evidence=""):
        self.passed = passed
        self.evidence = evidence


def _mastered_diagnosis():
    return Diagnosis(
        point_id="aop-concept",
        state=CognitiveState.MASTERED,
        confidence=Confidence.HIGH,
        evidence=["用户准确解释了切面与连接点"],
    )


def _partial_diagnosis():
    return Diagnosis(
        point_id="aop-concept",
        state=CognitiveState.PARTIAL,
        confidence=Confidence.HIGH,
        evidence=["能说大意但边界模糊"],
    )


def _cfg(user_id="u1"):
    """构造带 user_id 的 runtime config（宪法 §5：user_id 走 runtime context）。"""
    return {"configurable": {"user_id": user_id}}


def _make_tools(diagnoser_responses, teacher_responses=None, store=None):
    """构造工具集：diagnoser / teacher 用 ScriptedLLM，planner 用空桩，store 注入。

    user_id 不再经闭包注入，由调用工具时通过 runtime config 传入（宪法 §5）。
    """
    diagnoser = ScriptedLLM(diagnoser_responses)
    teacher = ScriptedLLM(teacher_responses or [])
    planner = ScriptedLLM([])
    return build_cognia_tools(
        diagnoser=diagnoser,
        planner=planner,
        teacher=teacher,
        store=store or InMemoryStore(),
    )


# ---- 读工具 ----

def test_read_learner_state():
    """读工具返回当前五态（未评估时为 unassessed）。"""
    from cognia.memory import append_proficiency_delta
    from cognia.schemas import ProficiencyEntry
    from datetime import datetime, timezone

    store = InMemoryStore()
    append_proficiency_delta(store, "u1", ProficiencyEntry(
        point_id="aop-concept",
        from_state=None,
        to_state=CognitiveState.PARTIAL,
        evidence=["e"],
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
    ))
    tools = build_cognia_tools(store=store)

    assert tools["read_learner_state"].invoke(
        {"point_id": "aop-concept"}, config=_cfg()
    ) == '{"state": "partial"}'
    assert tools["read_learner_state"].invoke(
        {"point_id": "unknown-point"}, config=_cfg()
    ) == '{"state": "unassessed"}'


# ---- 写工具：三层闸门 ----

def test_propose_diagnosis_medium_confidence_no_migration():
    """中置信度诊断不迁移（诊断 ≠ 迁移，三层闸门第一层）。"""
    diagnoser = ScriptedLLM([
        Diagnosis(point_id="aop-concept", state=CognitiveState.PARTIAL,
                  confidence=Confidence.MEDIUM, evidence=["模糊"]),
    ])
    store = InMemoryStore()
    tools = build_cognia_tools(diagnoser=diagnoser, store=store)

    result = json.loads(tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "大概是切面吧",
        "current_state": "unassessed",
    }, config=_cfg()))

    assert result["migrated"] is False
    assert result["final_state"] == "unassessed"

    from cognia.memory import get_current_proficiency
    assert get_current_proficiency(store, "u1", "aop-concept") is None  # 未写 delta


def test_propose_diagnosis_mastered_verification_fail_no_migration():
    """伪 mastered（验证失败）不迁移，诊断降级 partial，零 Delta。"""
    diagnoser = ScriptedLLM([
        _mastered_diagnosis(),
        _ConceptAssessmentStub(False, ""),
        _ScenarioAssessmentStub(False, ""),
    ])
    store = InMemoryStore()
    tools = build_cognia_tools(diagnoser=diagnoser, store=store)

    result = json.loads(tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "AOP 就是切面",
        "current_state": "partial",
    }, config=_cfg()))

    assert result["migrated"] is False
    assert result["final_state"] == "partial"          # 保持原状态
    assert result["diagnosed_state"] == "partial"       # 诊断已降级
    assert result["verification"] == {"concept": "failed", "scenario": "failed"}

    from cognia.memory import get_current_proficiency
    assert get_current_proficiency(store, "u1", "aop-concept") is None  # 未写 delta


def test_propose_diagnosis_mastered_with_verification_migrates():
    """mastered + 双重验证全过 → 迁移并增量写回 store。"""
    diagnoser = ScriptedLLM([
        _mastered_diagnosis(),
        _ConceptAssessmentStub(True, "概念正确"),
        _ScenarioAssessmentStub(True, "场景正确"),
    ])
    store = InMemoryStore()
    tools = build_cognia_tools(diagnoser=diagnoser, store=store)

    result = json.loads(tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "AOP 通过切面拦截方法调用",
        "current_state": "partial",
    }, config=_cfg()))

    assert result["migrated"] is True
    assert result["final_state"] == "mastered"
    assert result["verification"] == {"concept": "passed", "scenario": "passed"}

    from cognia.memory import get_current_proficiency
    assert get_current_proficiency(store, "u1", "aop-concept") == "mastered"  # 已写 delta


def test_propose_diagnosis_partial_high_confidence_migrates():
    """高置信度 partial（非 mastered）经状态机裁决迁移，无需双重验证。"""
    diagnoser = ScriptedLLM([_partial_diagnosis()])
    store = InMemoryStore()
    tools = build_cognia_tools(diagnoser=diagnoser, store=store)

    result = json.loads(tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "AOP 就是切面，但代理机制说不清",
        "current_state": "unassessed",
    }, config=_cfg()))

    assert result["migrated"] is True
    assert result["final_state"] == "partial"

    from cognia.memory import get_current_proficiency
    assert get_current_proficiency(store, "u1", "aop-concept") == "partial"


# ---- 教学工具 ----

def test_generate_probe():
    """generate_probe 生成探针问题文本。"""
    tools = _make_tools(
        diagnoser_responses=[],
        teacher_responses=[_Msg("请用自己的话解释什么是 AOP？")],
    )
    result = tools["generate_probe"].invoke({
        "point_name": "AOP 概念", "point_description": "面向切面编程",
    })
    assert result == "请用自己的话解释什么是 AOP？"


def test_explain():
    """explain 针对认知状态生成讲解文本。"""
    tools = _make_tools(
        diagnoser_responses=[],
        teacher_responses=[_Msg("AOP 允许你把横切关注点抽离成切面……")],
    )
    result = tools["explain"].invoke({
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "user_state": "partial",
    })
    assert "AOP" in result


# ---- 知识模型 load-or-build ----

def test_build_learning_goal_builds_then_reuses():
    """首次构建知识模型并冻结，第二次复用（planner 不再被调用）。"""
    planner = ScriptedLLM([
        KnowledgeModel(goal="Spring AOP", points=[
            KnowledgePoint(id="aop-concept", name="AOP 概念", description="面向切面编程"),
        ]),
    ])
    store = InMemoryStore()
    tools = build_cognia_tools(
        diagnoser=ScriptedLLM([]),
        planner=planner,
        teacher=ScriptedLLM([]),
        store=store,
    )

    first = json.loads(tools["build_learning_goal"].invoke({"goal": "Spring AOP"}, config=_cfg()))
    assert first["action"] == "构建"
    assert first["first_point_id"] == "aop-concept"

    # 第二次：复用，planner 不再被调用（队列已空，若重建会 assert 耗尽）
    second = json.loads(tools["build_learning_goal"].invoke({"goal": "Spring AOP"}, config=_cfg()))
    assert second["action"] == "复用"
    assert second["first_point_id"] == "aop-concept"


# ---- 写工具：record_observation（观察样本，只追加不改结论）----

def test_record_observation_appends_only():
    """record_observation 只追加观察，不直接写 proficiency 结论。"""
    store = InMemoryStore()
    tools = build_cognia_tools(store=store)

    result = json.loads(tools["record_observation"].invoke({
        "point_id": "aop-concept",
        "observed_state": "partial",
        "confidence": "high",
        "evidence": ["用户说 AOP 是切面，但说不清代理"],
    }, config=_cfg()))

    assert result["recorded"] is True

    from cognia.memory import query_observations, get_current_proficiency
    obs = query_observations(store, "u1", "aop-concept")
    assert len(obs) == 1
    assert obs[0]["observed_state"] == "partial"
    # 关键：没有直接写 proficiency Delta（AI 不能直接改结论）
    assert get_current_proficiency(store, "u1", "aop-concept") is None


def test_record_observation_unassessed_skipped():
    """unassessed 不产生观测。"""
    store = InMemoryStore()
    tools = build_cognia_tools(store=store)

    result = json.loads(tools["record_observation"].invoke({
        "point_id": "aop-concept",
        "observed_state": "unassessed",
        "confidence": "high",
        "evidence": ["x"],
    }, config=_cfg()))

    assert result["recorded"] is False

    from cognia.memory import query_observations
    assert query_observations(store, "u1", "aop-concept") == []


def test_record_observation_empty_evidence_rejected():
    """空 evidence 拒绝写入。"""
    store = InMemoryStore()
    tools = build_cognia_tools(store=store)

    result = json.loads(tools["record_observation"].invoke({
        "point_id": "aop-concept",
        "observed_state": "partial",
        "confidence": "high",
        "evidence": [],
    }, config=_cfg()))

    assert result["recorded"] is False


def test_record_observation_invalid_state():
    """非法 observed_state 返回错误，不落库。"""
    store = InMemoryStore()
    tools = build_cognia_tools(store=store)

    result = json.loads(tools["record_observation"].invoke({
        "point_id": "aop-concept",
        "observed_state": "nonsense",
        "confidence": "high",
        "evidence": ["x"],
    }, config=_cfg()))

    assert result["recorded"] is False
    assert "error" in result


def test_record_observation_invalid_confidence():
    """非法 confidence 返回错误。"""
    store = InMemoryStore()
    tools = build_cognia_tools(store=store)

    result = json.loads(tools["record_observation"].invoke({
        "point_id": "aop-concept",
        "observed_state": "partial",
        "confidence": "nonsense",
        "evidence": ["x"],
    }, config=_cfg()))

    assert result["recorded"] is False
    assert "error" in result


def test_record_observation_store_none_degrades():
    """store 为 None 时安全降级，recorded=False。"""
    tools = build_cognia_tools(store=None)

    result = json.loads(tools["record_observation"].invoke({
        "point_id": "aop-concept",
        "observed_state": "partial",
        "confidence": "high",
        "evidence": ["x"],
    }, config=_cfg()))

    assert result["recorded"] is False


# ---- 读工具：query_proficiency（查询权威状态）----

def test_query_proficiency_unassessed():
    """无观察 → 返回 unassessed，latent_value 为 BKT 先验 0.4，observation_count=0。"""
    store = InMemoryStore()
    tools = build_cognia_tools(store=store)

    result = json.loads(tools["query_proficiency"].invoke(
        {"point_id": "aop-concept"}, config=_cfg()
    ))

    assert result["mapped_state"] == "unassessed"
    assert result["latent_value"] == 0.4  # BKT 先验 P(L0)
    assert result["observation_count"] == 0


def test_query_proficiency_with_observation():
    """有观察 → 返回系统 BKT 融合后的权威状态。"""
    store = InMemoryStore()
    tools = build_cognia_tools(store=store)

    tools["record_observation"].invoke({
        "point_id": "aop-concept",
        "observed_state": "mastered",
        "confidence": "high",
        "evidence": ["用户准确解释切面"],
    }, config=_cfg())

    result = json.loads(tools["query_proficiency"].invoke(
        {"point_id": "aop-concept"}, config=_cfg()
    ))

    assert result["point_id"] == "aop-concept"
    assert result["mapped_state"] in ("mastered", "partial")
    assert result["latent_value"] is not None
    assert 0.0 <= result["latent_value"] <= 1.0
    assert result["observation_count"] == 1


def test_query_proficiency_store_none_degrades():
    """store 为 None → 返回 unassessed 降级（不抛错）。"""
    tools = build_cognia_tools(store=None)

    result = json.loads(tools["query_proficiency"].invoke(
        {"point_id": "aop-concept"}, config=_cfg()
    ))

    assert result["mapped_state"] == "unassessed"


def test_query_proficiency_reads_authoritative_not_raw():
    """查询的是系统权威状态，不是 AI 观察值本身（latent_value 为 BKT 后验）。"""
    store = InMemoryStore()
    tools = build_cognia_tools(store=store)

    # 提交一次 misconception 观察：AI 观察值是 misconception，但系统权威状态
    # 由 BKT 后验 + 离散化得出，mapped_state 应为 misconception（负向观察优先）。
    tools["record_observation"].invoke({
        "point_id": "aop-concept",
        "observed_state": "misconception",
        "confidence": "high",
        "evidence": ["用户认为 @Around 修改字节码"],
    }, config=_cfg())

    result = json.loads(tools["query_proficiency"].invoke(
        {"point_id": "aop-concept"}, config=_cfg()
    ))

    assert result["mapped_state"] == "misconception"
    # latent_value 是连续概率（0-1），不是观察值本身
    assert 0.0 <= result["latent_value"] <= 1.0
