"""cognia.graph 教学核心图单元测试。

全部使用 ScriptedLLM 假模型，离线、快速、不依赖 DEEPSEEK_API_KEY。
重点验证三类验收标准：
1. mock 跑通最小闭环（mastered 闭环）
2. 中/低置信度诊断不修改长期认知状态（诊断 ≠ 迁移）
3. 3 轮失败回溯 + mastered 双重验证闸门
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from cognia.graph import build_graph, resolve_migration
from cognia.schemas import (
    CognitiveState,
    Confidence,
    Diagnosis,
    KnowledgeModel,
    KnowledgePoint,
    ValidationResult,
    VerificationState,
)


class ScriptedLLM:
    """按脚本队列返回预设响应的假模型。

    支持 with_structured_output（返回 self）与 invoke（按队列弹出响应）。
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


def _point(point_id="aop-concept", name="AOP 概念"):
    return KnowledgePoint(id=point_id, name=name, description="面向切面编程")


def _single_point_model():
    return KnowledgeModel(goal="Spring AOP", points=[_point()])


# ---- 纯函数：resolve_migration（诊断 ≠ 迁移 的核心）----

def test_resolve_migration_medium_confidence_no_migration():
    """中置信度诊断不产生迁移。"""
    d = Diagnosis(point_id="p", state=CognitiveState.PARTIAL, confidence=Confidence.MEDIUM, evidence=["x"])
    assert resolve_migration(d, None, None) is None


def test_resolve_migration_low_confidence_no_migration():
    """低置信度诊断不产生迁移。"""
    d = Diagnosis(point_id="p", state=CognitiveState.PARTIAL, confidence=Confidence.LOW, evidence=["x"])
    assert resolve_migration(d, None, None) is None


def test_resolve_migration_mastered_requires_verification():
    """高置信度 mastered 候选，但双重验证未过 → 不迁移。"""
    d = Diagnosis(point_id="p", state=CognitiveState.MASTERED, confidence=Confidence.HIGH, evidence=["x"])
    v = VerificationState(concept=ValidationResult.PASSED, scenario=ValidationResult.UNASSESSED)
    assert resolve_migration(d, CognitiveState.PARTIAL, v) is None


def test_resolve_migration_mastered_with_verification():
    """高置信度 mastered + 双重验证全过 → 迁移。"""
    d = Diagnosis(point_id="p", state=CognitiveState.MASTERED, confidence=Confidence.HIGH, evidence=["x"])
    v = VerificationState(concept=ValidationResult.PASSED, scenario=ValidationResult.PASSED)
    entry = resolve_migration(d, CognitiveState.PARTIAL, v)
    assert entry is not None
    assert entry.to_state == CognitiveState.MASTERED
    assert entry.from_state == CognitiveState.PARTIAL


def test_resolve_migration_self_transition_forbidden():
    """自我迁移（partial→partial）被拓扑禁止 → 不迁移。"""
    d = Diagnosis(point_id="p", state=CognitiveState.PARTIAL, confidence=Confidence.HIGH, evidence=["x"])
    assert resolve_migration(d, CognitiveState.PARTIAL, None) is None


def test_resolve_migration_first_diagnosis_partial_allowed():
    """首次诊断（from=None）partial 高置信度 → 迁移（unassessed→partial）。"""
    d = Diagnosis(point_id="p", state=CognitiveState.PARTIAL, confidence=Confidence.HIGH, evidence=["x"])
    entry = resolve_migration(d, None, None)
    assert entry is not None
    assert entry.from_state is None
    assert entry.to_state == CognitiveState.PARTIAL


# ---- 图集成测试 ----

def _resume_times(graph, config, answers):
    """按顺序 resume，返回最后一次结果。"""
    result = None
    for ans in answers:
        result = graph.invoke(Command(resume=ans), config=config)
    return result


def test_minimal_closed_loop_mastered():
    """单知识点 mastered 最小闭环：设目标→建模型→探测→诊断→验证→掌握→结束。"""
    planner = ScriptedLLM([
        _GoalAssessmentStub(False),   # setup_goal：目标不过大
        _single_point_model(),        # build_model：单知识点
    ])
    teacher = ScriptedLLM(["请解释 AOP 是什么？"])  # probe
    diagnoser = ScriptedLLM([
        Diagnosis(point_id="aop-concept", state=CognitiveState.MASTERED,
                  confidence=Confidence.HIGH, evidence=["用户准确解释了切面与连接点"]),
        _ConceptAssessmentStub(True, "概念解释正确"),
        _ScenarioAssessmentStub(True, "用户能辨析动态代理与 CGLIB 的边界"),
    ])

    graph = build_graph(planner_model=planner, teacher_model=teacher,
                        diagnoser_model=diagnoser, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1"}}

    first = graph.invoke({"goal": "Spring AOP"}, config=config)
    assert "__interrupt__" in first  # 第一次中断在 await_answer

    final = _resume_times(graph, config, ["AOP 是面向切面编程，通过切面拦截方法调用……"])

    assert final["ended"] is True
    assert len(final["proficiency_deltas"]) == 1
    assert final["proficiency_deltas"][0].to_state == CognitiveState.MASTERED
    assert final["proficiency_deltas"][0].from_state is None


def test_medium_confidence_no_delta():
    """中置信度诊断不产生迁移（诊断 ≠ 迁移，图集成验证）。"""
    planner = ScriptedLLM([
        _GoalAssessmentStub(False),
        _single_point_model(),
    ])
    teacher = ScriptedLLM(["q1", "q2"])  # 两轮 probe
    diagnoser = ScriptedLLM([
        Diagnosis(point_id="aop-concept", state=CognitiveState.PARTIAL,
                  confidence=Confidence.MEDIUM, evidence=["说得有点模糊"]),
        Diagnosis(point_id="aop-concept", state=CognitiveState.PARTIAL,
                  confidence=Confidence.HIGH, evidence=["说清楚了部分"]),
    ])

    graph = build_graph(planner_model=planner, teacher_model=teacher,
                        diagnoser_model=diagnoser, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t2"}}

    graph.invoke({"goal": "Spring AOP"}, config=config)  # 中断在第一次 await
    graph.invoke(Command(resume="AOP 就是切面"), config=config)  # 中置信度 → 路由回 probe → 再次中断

    snapshot = graph.get_state(config)
    state = snapshot.values
    assert state["diagnosis"].confidence == Confidence.MEDIUM
    assert state.get("proficiency_deltas", []) == []  # 中置信度不迁移


def test_three_failures_backtrack():
    """3 轮干预失败（仍 misconception）触发回溯，放弃当前知识点。"""
    planner = ScriptedLLM([
        _GoalAssessmentStub(False),
        _single_point_model(),
    ])
    teacher = ScriptedLLM([
        "q1", "i1", "q2", "i2", "q3", "i3", "q4",  # 4 次 probe + 3 次 intervene
    ])
    diagnoser = ScriptedLLM([
        Diagnosis(point_id="aop-concept", state=CognitiveState.MISCONCEPTION,
                  confidence=Confidence.HIGH, evidence=["错误"]),
        Diagnosis(point_id="aop-concept", state=CognitiveState.MISCONCEPTION,
                  confidence=Confidence.HIGH, evidence=["错误"]),
        Diagnosis(point_id="aop-concept", state=CognitiveState.MISCONCEPTION,
                  confidence=Confidence.HIGH, evidence=["错误"]),
        Diagnosis(point_id="aop-concept", state=CognitiveState.MISCONCEPTION,
                  confidence=Confidence.HIGH, evidence=["错误"]),
    ])

    graph = build_graph(planner_model=planner, teacher_model=teacher,
                        diagnoser_model=diagnoser, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t3"}}

    graph.invoke({"goal": "Spring AOP"}, config=config)
    final = _resume_times(graph, config, ["答1", "答2", "答3", "答4"])

    assert final["ended"] is True
    assert final["intervention_fail_count"] == 3
    # 仅首次 unassessed→misconception 产生 1 个 Delta，后续自我迁移被禁止
    assert len(final["proficiency_deltas"]) == 1
    assert final["proficiency_deltas"][0].to_state == CognitiveState.MISCONCEPTION


def test_verify_failure_backtrack():
    """伪 mastered（diagnose 判 mastered 但 verify 反复失败）→ 3 次后回溯，防无限循环。

    这是反馈发现的核心 bug：若 verify 失败不计数，_improved(mastered, ...) 恒 True
    会导致「diagnose→verify 失败→intervene→probe→diagnose」无限循环。
    """
    planner = ScriptedLLM([
        _GoalAssessmentStub(False),
        _single_point_model(),
    ])
    teacher = ScriptedLLM([
        "q1", "i1", "q2", "i2", "q3", "i3", "q4",
    ])
    # 4 次 diagnose 均判 mastered；前 3 次 verify（concept+scenario）均失败；
    # 第 4 次 resume 时 fail_count=3 命中闸门 → 直接 select_next，不再 verify。
    diagnoser = ScriptedLLM([
        Diagnosis(point_id="aop-concept", state=CognitiveState.MASTERED,
                  confidence=Confidence.HIGH, evidence=["说得对"]),
        _ConceptAssessmentStub(False, "概念不清"),
        _ScenarioAssessmentStub(False, "场景错误"),
        Diagnosis(point_id="aop-concept", state=CognitiveState.MASTERED,
                  confidence=Confidence.HIGH, evidence=["说得对"]),
        _ConceptAssessmentStub(False, "概念不清"),
        _ScenarioAssessmentStub(False, "场景错误"),
        Diagnosis(point_id="aop-concept", state=CognitiveState.MASTERED,
                  confidence=Confidence.HIGH, evidence=["说得对"]),
        _ConceptAssessmentStub(False, "概念不清"),
        _ScenarioAssessmentStub(False, "场景错误"),
        Diagnosis(point_id="aop-concept", state=CognitiveState.MASTERED,
                  confidence=Confidence.HIGH, evidence=["说得对"]),
    ])

    graph = build_graph(planner_model=planner, teacher_model=teacher,
                        diagnoser_model=diagnoser, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t4"}}

    graph.invoke({"goal": "Spring AOP"}, config=config)
    final = _resume_times(graph, config, ["答1", "答2", "答3", "答4"])

    assert final["ended"] is True
    assert final["intervention_fail_count"] == 3
    # mastered 从未通过双重验证，故零 Delta
    assert final.get("proficiency_deltas", []) == []


def test_verify_failure_downgrades_diagnosis():
    """verify 失败后 diagnosis.state 降级到 partial，且 concept_evidence 不回填 mastered 证据。

    反馈发现的遗留问题：verify 失败若不改 diagnosis，intervene 会拿到「已 mastered +
    请干预」的自相矛盾状态；同时 concept 失败时 concept_evidence 曾错误回填
    diagnosis.evidence（支持「判 mastered」的正面证据）。
    """
    planner = ScriptedLLM([
        _GoalAssessmentStub(False),
        _single_point_model(),
    ])
    teacher = ScriptedLLM([
        "q1",  # 第一次 probe
        "i1",  # intervene（verify 失败后）
        "q2",  # 第二次 probe
    ])
    diagnoser = ScriptedLLM([
        Diagnosis(point_id="aop-concept", state=CognitiveState.MASTERED,
                  confidence=Confidence.HIGH, evidence=["正面证据"]),
        _ConceptAssessmentStub(False, ""),   # concept 失败 + 空 evidence
        _ScenarioAssessmentStub(False, ""),  # scenario 失败 + 空 evidence
    ])

    graph = build_graph(planner_model=planner, teacher_model=teacher,
                        diagnoser_model=diagnoser, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t5"}}

    graph.invoke({"goal": "Spring AOP"}, config=config)  # 中断在第一次 await
    graph.invoke(Command(resume="答案"), config=config)  # diagnose→verify失败→intervene→probe→中断

    snapshot = graph.get_state(config)
    state = snapshot.values
    assert state["diagnosis"].state == CognitiveState.PARTIAL  # 已降级
    assert state["diagnosis"].confidence == Confidence.HIGH    # 置信度不变
    assert state["verification"].concept == ValidationResult.FAILED
    assert state["verification"].scenario == ValidationResult.FAILED
    assert state["intervention_fail_count"] == 1
    assert state["verification"].concept_evidence == []  # 不回填 mastered 正面证据


def test_intervene_reads_verification_gap():
    """intervene 消费 verification 结果，针对「场景薄弱」做针对性纠错（而非笼统干预）。

    反馈的可选改进：verify 失败后应让 intervene 知道「概念 vs 场景」哪块没过，
    才能针对性纠错。此测试用 scenario 失败 + concept 通过的组合验证区分能力。
    """
    planner = ScriptedLLM([
        _GoalAssessmentStub(False),
        _single_point_model(),
    ])
    teacher = ScriptedLLM([
        "q1",  # 第一次 probe
        "i1",  # intervene（verify 失败后）
        "q2",  # 第二次 probe
    ])
    diagnoser = ScriptedLLM([
        Diagnosis(point_id="aop-concept", state=CognitiveState.MASTERED,
                  confidence=Confidence.HIGH, evidence=["正面证据"]),
        _ConceptAssessmentStub(True, "概念正确"),   # concept 通过
        _ScenarioAssessmentStub(False, ""),        # scenario 失败
    ])

    graph = build_graph(planner_model=planner, teacher_model=teacher,
                        diagnoser_model=diagnoser, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t6"}}

    graph.invoke({"goal": "Spring AOP"}, config=config)  # 中断在第一次 await
    graph.invoke(Command(resume="答案"), config=config)  # diagnose→verify失败→intervene→probe→中断

    # teacher.calls[1] 是 intervene 的 prompt（list of (role, content)）
    intervene_messages = teacher.calls[1]
    human_content = intervene_messages[1][1]
    assert "场景/边界辨析未通过" in human_content
    assert "概念解释未通过" not in human_content  # 概念已通过，不应提示概念薄弱


# ---- 测试内部桩对象（代替真实 Pydantic 私有模型，避免 import 私有类）----

class _GoalAssessmentStub:
    """setup_goal 用 planner.with_structured_output 返回的桩对象。"""

    def __init__(self, too_broad, feedback=""):
        self.too_broad = too_broad
        self.feedback = feedback


class _ScenarioAssessmentStub:
    """verify 用 diagnoser.with_structured_output 返回的桩对象。"""

    def __init__(self, passed, evidence=""):
        self.passed = passed
        self.evidence = evidence


class _ConceptAssessmentStub:
    """verify 的概念验证用 diagnoser.with_structured_output 返回的桩对象。"""

    def __init__(self, passed, evidence=""):
        self.passed = passed
        self.evidence = evidence
