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


def _make_tools(diagnoser_responses, teacher_responses=None, store=None, user_id="u1"):
    """构造工具集：diagnoser / teacher 用 ScriptedLLM，planner 用空桩，store / user_id 注入。"""
    diagnoser = ScriptedLLM(diagnoser_responses)
    teacher = ScriptedLLM(teacher_responses or [])
    planner = ScriptedLLM([])
    return build_cognia_tools(
        diagnoser=diagnoser,
        planner=planner,
        teacher=teacher,
        store=store or InMemoryStore(),
        user_id=user_id,
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
    tools = build_cognia_tools(store=store, user_id="u1")

    assert tools["read_learner_state"].invoke({"point_id": "aop-concept"}) == '{"state": "partial"}'
    assert tools["read_learner_state"].invoke({"point_id": "unknown-point"}) == '{"state": "unassessed"}'


# ---- 写工具：三层闸门 ----

def test_propose_diagnosis_medium_confidence_no_migration():
    """中置信度诊断不迁移（诊断 ≠ 迁移，三层闸门第一层）。"""
    diagnoser = ScriptedLLM([
        Diagnosis(point_id="aop-concept", state=CognitiveState.PARTIAL,
                  confidence=Confidence.MEDIUM, evidence=["模糊"]),
    ])
    store = InMemoryStore()
    tools = build_cognia_tools(diagnoser=diagnoser, store=store, user_id="u1")

    result = json.loads(tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "大概是切面吧",
        "current_state": "unassessed",
    }))

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
    tools = build_cognia_tools(diagnoser=diagnoser, store=store, user_id="u1")

    result = json.loads(tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "AOP 就是切面",
        "current_state": "partial",
    }))

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
    tools = build_cognia_tools(diagnoser=diagnoser, store=store, user_id="u1")

    result = json.loads(tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "AOP 通过切面拦截方法调用",
        "current_state": "partial",
    }))

    assert result["migrated"] is True
    assert result["final_state"] == "mastered"
    assert result["verification"] == {"concept": "passed", "scenario": "passed"}

    from cognia.memory import get_current_proficiency
    assert get_current_proficiency(store, "u1", "aop-concept") == "mastered"  # 已写 delta


def test_propose_diagnosis_partial_high_confidence_migrates():
    """高置信度 partial（非 mastered）经状态机裁决迁移，无需双重验证。"""
    diagnoser = ScriptedLLM([_partial_diagnosis()])
    store = InMemoryStore()
    tools = build_cognia_tools(diagnoser=diagnoser, store=store, user_id="u1")

    result = json.loads(tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "AOP 就是切面，但代理机制说不清",
        "current_state": "unassessed",
    }))

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
        user_id="u1",
    )

    first = json.loads(tools["build_learning_goal"].invoke({"goal": "Spring AOP"}))
    assert first["action"] == "构建"
    assert first["first_point_id"] == "aop-concept"

    # 第二次：复用，planner 不再被调用（队列已空，若重建会 assert 耗尽）
    second = json.loads(tools["build_learning_goal"].invoke({"goal": "Spring AOP"}))
    assert second["action"] == "复用"
    assert second["first_point_id"] == "aop-concept"
