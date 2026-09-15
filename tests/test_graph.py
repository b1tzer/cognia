"""cognia.graph 编排层单元测试（重构后：对话 Agent ⇄ 学习引擎循环）。

全部使用 ScriptedLLM 假模型，离线、快速、不依赖 DEEPSEEK_API_KEY。

重点验证新架构的四类不变量：
1. **对话自主权与认知裁决权解耦**：Agent 可自由决定怎么聊 / 怎么教，
   但 ConversationTurn 里没有认知状态字段；状态只能经 Learning Engine 变更。
2. **诊断 ≠ 迁移**：中 / 低置信度诊断不修改长期认知状态；迁移必须经状态机裁决。
3. **掌握双重验证**：diagnose 判 mastered 只是候选，必须概念 + 场景双过才迁移。
4. **防失控**：3 轮干预失败 / 连续无进展触发放弃（回溯），跨会话读回历史熟练度。
"""

from datetime import datetime, timezone

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from cognia.conversation_agent import (
    ConversationAction,
    ConversationIntent,
    ConversationTurn,
)
from cognia.graph import build_graph, resolve_migration
from cognia.memory import append_proficiency_delta
from cognia.schemas import (
    CognitiveState,
    Confidence,
    Diagnosis,
    KnowledgeModel,
    KnowledgePoint,
    ProficiencyEntry,
)


class ScriptedLLM:
    """按脚本队列返回预设响应的假模型。

    支持 with_structured_output（返回 self）与 invoke（按队列弹出响应），
    覆盖三种角色：对话 Agent（返回 ConversationTurn）、planner（返回 KnowledgeModel）、
    diagnoser（返回 Diagnosis / 验证桩对象）。
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


# ---- 测试桩 / 助手 ----

def _turn(intent, action, reply="", proposed_goal=None, target_point_id=None):
    """构造一轮对话 Agent 的结构化决策。"""
    return ConversationTurn(
        intent=intent,
        action=action,
        reply=reply,
        proposed_goal=proposed_goal,
        target_point_id=target_point_id,
    )


def _point(point_id="aop-concept", name="AOP 概念"):
    return KnowledgePoint(id=point_id, name=name, description="面向切面编程")


def _single_point_model():
    return KnowledgeModel(goal="Spring AOP", points=[_point()])


def _two_point_model():
    return KnowledgeModel(goal="Spring AOP", points=[
        _point("aop-concept", "AOP 概念"),
        _point("aop-proxy", "AOP 代理"),
    ])


class _ConceptAssessmentStub:
    def __init__(self, passed, evidence=""):
        self.passed = passed
        self.evidence = evidence


class _ScenarioAssessmentStub:
    def __init__(self, passed, evidence=""):
        self.passed = passed
        self.evidence = evidence


def _mastered_diagnosis(point_id="aop-concept"):
    return Diagnosis(
        point_id=point_id,
        state=CognitiveState.MASTERED,
        confidence=Confidence.HIGH,
        evidence=["用户准确解释了切面与连接点"],
    )


# ---- 纯函数：resolve_migration（诊断 ≠ 迁移 的核心）----

def test_resolve_migration_medium_confidence_no_migration():
    d = Diagnosis(point_id="p", state=CognitiveState.PARTIAL, confidence=Confidence.MEDIUM, evidence=["x"])
    assert resolve_migration(d, None, None) is None


def test_resolve_migration_low_confidence_no_migration():
    d = Diagnosis(point_id="p", state=CognitiveState.PARTIAL, confidence=Confidence.LOW, evidence=["x"])
    assert resolve_migration(d, None, None) is None


def test_resolve_migration_mastered_requires_verification():
    from cognia.schemas import ValidationResult, VerificationState
    d = Diagnosis(point_id="p", state=CognitiveState.MASTERED, confidence=Confidence.HIGH, evidence=["x"])
    v = VerificationState(concept=ValidationResult.PASSED, scenario=ValidationResult.UNASSESSED)
    assert resolve_migration(d, CognitiveState.PARTIAL, v) is None


def test_resolve_migration_mastered_with_verification():
    from cognia.schemas import ValidationResult, VerificationState
    d = Diagnosis(point_id="p", state=CognitiveState.MASTERED, confidence=Confidence.HIGH, evidence=["x"])
    v = VerificationState(concept=ValidationResult.PASSED, scenario=ValidationResult.PASSED)
    entry = resolve_migration(d, CognitiveState.PARTIAL, v)
    assert entry is not None
    assert entry.to_state == CognitiveState.MASTERED
    assert entry.from_state == CognitiveState.PARTIAL


def test_resolve_migration_self_transition_forbidden():
    d = Diagnosis(point_id="p", state=CognitiveState.PARTIAL, confidence=Confidence.HIGH, evidence=["x"])
    assert resolve_migration(d, CognitiveState.PARTIAL, None) is None


def test_resolve_migration_first_diagnosis_partial_allowed():
    d = Diagnosis(point_id="p", state=CognitiveState.PARTIAL, confidence=Confidence.HIGH, evidence=["x"])
    entry = resolve_migration(d, None, None)
    assert entry is not None
    assert entry.from_state is None
    assert entry.to_state == CognitiveState.PARTIAL


# ---- 结构不变量：Agent 无认知裁决权 ----

def test_agent_turn_has_no_cognitive_state_field():
    """ConversationTurn 不得包含任何认知状态字段（Agent 只能请求诊断，不能直写）。"""
    fields = set(ConversationTurn.model_fields.keys())
    assert "state" not in fields
    assert "mastery" not in fields
    assert "cognitive_state" not in fields
    assert "proficiency" not in fields


# ---- 图集成：对话自主权（不把输入强行解释成节点要求的格式）----

def test_greeting_not_forced_as_goal():
    """用户「你好」→ Agent 自然回应，不建知识模型、不设学习目标。"""
    agent = ScriptedLLM([
        _turn(ConversationIntent.GREETING, ConversationAction.RESPOND,
              reply="你好！今天想学点什么？"),
    ])
    graph = build_graph(
        planner_model=ScriptedLLM([]),
        teacher_model=agent,
        diagnoser_model=ScriptedLLM([]),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-greet"}}

    result = graph.invoke({"user_message": "你好"}, config=config)

    assert "__interrupt__" in result
    interrupt_value = result["__interrupt__"][0].value
    assert interrupt_value["message"] == "你好！今天想学点什么？"

    state = graph.get_state(config).values
    assert state.get("knowledge_model") is None
    assert state.get("goal") is None


def test_set_goal_builds_model_and_probes():
    """用户「Spring AOP」→ Agent 判为 SET_GOAL → 引擎建模 → Agent 提出探针问题。"""
    planner = ScriptedLLM([_single_point_model()])
    agent = ScriptedLLM([
        _turn(ConversationIntent.SET_GOAL, ConversationAction.SET_GOAL,
              proposed_goal="Spring AOP"),
        _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="请解释一下 AOP 是什么？"),
    ])
    graph = build_graph(
        planner_model=planner,
        teacher_model=agent,
        diagnoser_model=ScriptedLLM([]),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-goal"}}

    result = graph.invoke({"user_message": "Spring AOP"}, config=config)

    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["message"] == "请解释一下 AOP 是什么？"

    state = graph.get_state(config).values
    assert state["goal"] == "Spring AOP"
    assert state["current_point_id"] == "aop-concept"
    assert state["knowledge_model"]["points"][0]["id"] == "aop-concept"


# ---- 图集成：mastered 闭环与诊断 ≠ 迁移 ----

def test_mastered_closed_loop():
    """单知识点 mastered 最小闭环：设目标 → 建模 → 探测 → 诊断 → 双重验证 → 掌握 → 结束。"""
    planner = ScriptedLLM([_single_point_model()])
    agent = ScriptedLLM([
        _turn(ConversationIntent.SET_GOAL, ConversationAction.SET_GOAL,
              proposed_goal="Spring AOP"),
        _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="请解释 AOP 是什么？"),
        _turn(ConversationIntent.ANSWER, ConversationAction.DIAGNOSE),
        _turn(ConversationIntent.ANSWER, ConversationAction.RESPOND, reply="恭喜你掌握了 AOP！"),
    ])
    diagnoser = ScriptedLLM([
        _mastered_diagnosis(),
        _ConceptAssessmentStub(True, "概念解释正确"),
        _ScenarioAssessmentStub(True, "用户能辨析动态代理与 CGLIB 的边界"),
    ])

    graph = build_graph(
        planner_model=planner,
        teacher_model=agent,
        diagnoser_model=diagnoser,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-mastered"}}

    first = graph.invoke({"user_message": "Spring AOP"}, config=config)
    assert "__interrupt__" in first

    final = graph.invoke(
        Command(resume="AOP 是面向切面编程，通过切面拦截方法调用……"),
        config=config,
    )

    assert final["ended"] is True
    assert len(final["proficiency_deltas"]) == 1
    assert final["proficiency_deltas"][0]["to_state"] == "mastered"
    assert final["proficiency_deltas"][0]["from_state"] is None


def test_medium_confidence_no_delta():
    """中置信度诊断不产生迁移（诊断 ≠ 迁移），Agent 重新探测。"""
    planner = ScriptedLLM([_single_point_model()])
    agent = ScriptedLLM([
        _turn(ConversationIntent.SET_GOAL, ConversationAction.SET_GOAL,
              proposed_goal="Spring AOP"),
        _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="q1"),
        _turn(ConversationIntent.ANSWER, ConversationAction.DIAGNOSE),
        _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="q2"),
    ])
    diagnoser = ScriptedLLM([
        Diagnosis(point_id="aop-concept", state=CognitiveState.PARTIAL,
                  confidence=Confidence.MEDIUM, evidence=["说得有点模糊"]),
    ])

    graph = build_graph(
        planner_model=planner,
        teacher_model=agent,
        diagnoser_model=diagnoser,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-medium"}}

    graph.invoke({"user_message": "Spring AOP"}, config=config)
    graph.invoke(Command(resume="AOP 就是切面"), config=config)

    state = graph.get_state(config).values
    assert state["diagnosis"]["confidence"] == "medium"
    assert state.get("proficiency_deltas", []) == []  # 中置信度不迁移


def test_verify_failure_downgrades_diagnosis():
    """伪 mastered（diagnose 判 mastered 但验证失败）→ 降级 partial，零 mastered Delta。"""
    planner = ScriptedLLM([_single_point_model()])
    agent = ScriptedLLM([
        _turn(ConversationIntent.SET_GOAL, ConversationAction.SET_GOAL,
              proposed_goal="Spring AOP"),
        _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="q1"),
        _turn(ConversationIntent.ANSWER, ConversationAction.DIAGNOSE),
        _turn(ConversationIntent.ANSWER, ConversationAction.EXPLAIN, reply="我再讲讲概念"),
    ])
    diagnoser = ScriptedLLM([
        _mastered_diagnosis(),
        _ConceptAssessmentStub(False, ""),
        _ScenarioAssessmentStub(False, ""),
    ])

    graph = build_graph(
        planner_model=planner,
        teacher_model=agent,
        diagnoser_model=diagnoser,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-verify-fail"}}

    graph.invoke({"user_message": "Spring AOP"}, config=config)
    graph.invoke(Command(resume="AOP 就是切面"), config=config)

    state = graph.get_state(config).values
    assert state["diagnosis"]["state"] == "partial"      # 已降级
    assert state["verification"]["concept"] == "failed"
    assert state["verification"]["scenario"] == "failed"
    assert state.get("proficiency_deltas", []) == []      # 未通过验证，零 mastered Delta


# ---- 图集成：防失控（3 轮失败回溯）----

def test_three_failures_backtrack():
    """3 轮干预失败（仍 misconception）触发放弃，且仅首次产生 1 个 Delta。"""
    planner = ScriptedLLM([_single_point_model()])
    agent = ScriptedLLM([
        _turn(ConversationIntent.SET_GOAL, ConversationAction.SET_GOAL,
              proposed_goal="Spring AOP"),
        _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="q1"),
        _turn(ConversationIntent.ANSWER, ConversationAction.DIAGNOSE),
        _turn(ConversationIntent.ANSWER, ConversationAction.EXPLAIN, reply="纠正一下"),
        _turn(ConversationIntent.ANSWER, ConversationAction.DIAGNOSE),
        _turn(ConversationIntent.ANSWER, ConversationAction.EXPLAIN, reply="再纠正"),
        _turn(ConversationIntent.ANSWER, ConversationAction.DIAGNOSE),
        _turn(ConversationIntent.ANSWER, ConversationAction.EXPLAIN, reply="再纠正"),
        _turn(ConversationIntent.ANSWER, ConversationAction.DIAGNOSE),
        _turn(ConversationIntent.ANSWER, ConversationAction.RESPOND, reply="暂时告一段落"),
    ])
    diagnoser = ScriptedLLM([
        Diagnosis(point_id="aop-concept", state=CognitiveState.MISCONCEPTION,
                  confidence=Confidence.HIGH, evidence=["错误"]) for _ in range(4)
    ])

    graph = build_graph(
        planner_model=planner,
        teacher_model=agent,
        diagnoser_model=diagnoser,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t-backtrack"}}

    graph.invoke({"user_message": "Spring AOP"}, config=config)
    final = None
    for ans in ["答1", "答2", "答3", "答4"]:
        final = graph.invoke(Command(resume=ans), config=config)

    assert final["ended"] is True
    assert final["intervention_fail_count"] == 3
    # 仅首次 unassessed→misconception 产生 1 个 Delta，后续自我迁移被禁止
    assert len(final["proficiency_deltas"]) == 1
    assert final["proficiency_deltas"][0]["to_state"] == "misconception"


# ---- 跨会话读侧闭环 ----

def test_cross_session_reads_historical_proficiency():
    """同一 user_id + 同一 point_id，SET_GOAL 建模时读回历史熟练度（而非 unassessed）。"""
    store = InMemoryStore()
    append_proficiency_delta(store, "u1", ProficiencyEntry(
        point_id="aop-concept",
        from_state=None,
        to_state=CognitiveState.PARTIAL,
        evidence=["历史证据"],
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
    ))

    planner = ScriptedLLM([_single_point_model()])
    agent = ScriptedLLM([
        _turn(ConversationIntent.SET_GOAL, ConversationAction.SET_GOAL,
              proposed_goal="Spring AOP"),
        _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="请解释 AOP"),
    ])
    diagnoser = ScriptedLLM([])

    graph = build_graph(
        planner_model=planner,
        teacher_model=agent,
        diagnoser_model=diagnoser,
        checkpointer=InMemorySaver(),
        store=store,
    )
    config = {"configurable": {"thread_id": "t-new", "user_id": "u1"}}

    graph.invoke({"user_message": "Spring AOP"}, config=config)

    state = graph.get_state(config).values
    assert state["current_long_state"] == "partial"  # 读回历史，而非 unassessed


def test_select_next_reads_historical_for_new_point():
    """掌握第一个点后切到第二个点，应读回该点历史态（而非硬置 None）。"""
    store = InMemoryStore()
    append_proficiency_delta(store, "u1", ProficiencyEntry(
        point_id="aop-proxy",
        from_state=None,
        to_state=CognitiveState.PARTIAL,
        evidence=["第二点历史"],
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
    ))

    planner = ScriptedLLM([_two_point_model()])
    agent = ScriptedLLM([
        _turn(ConversationIntent.SET_GOAL, ConversationAction.SET_GOAL,
              proposed_goal="Spring AOP"),
        _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="请解释 AOP 概念"),
        _turn(ConversationIntent.ANSWER, ConversationAction.DIAGNOSE),
        _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="请解释 AOP 代理"),
    ])
    diagnoser = ScriptedLLM([
        _mastered_diagnosis("aop-concept"),
        _ConceptAssessmentStub(True, "概念对"),
        _ScenarioAssessmentStub(True, "场景对"),
    ])

    graph = build_graph(
        planner_model=planner,
        teacher_model=agent,
        diagnoser_model=diagnoser,
        checkpointer=InMemorySaver(),
        store=store,
    )
    config = {"configurable": {"thread_id": "t-select", "user_id": "u1"}}

    graph.invoke({"user_message": "Spring AOP"}, config=config)
    graph.invoke(Command(resume="AOP 是面向切面"), config=config)

    state = graph.get_state(config).values
    assert state["current_point_id"] == "aop-proxy"
    assert state["current_long_state"] == "partial"  # 读回第二点历史
    assert len(state["proficiency_deltas"]) == 1
    assert state["proficiency_deltas"][0]["point_id"] == "aop-concept"
    assert state["proficiency_deltas"][0]["to_state"] == "mastered"


# ---- 知识模型 load-or-build（Task ⑨）----

def test_knowledge_model_reused_across_sessions():
    """同一 user + 同一 goal，第二次会话复用已冻结知识模型，不重新调用 planner。

    Task ⑨ 核心：point_id 由 LLM 每会话现生成会跨会话漂移（同一目标两次拆解
    得到不同 id），导致按 point_id 读回熟练度查空；load-or-build 冻结 km 后
    point_id 天然稳定。
    """
    store = InMemoryStore()

    # 第一次会话：构建并冻结知识模型
    graph1 = build_graph(
        planner_model=ScriptedLLM([_single_point_model()]),
        teacher_model=ScriptedLLM([
            _turn(ConversationIntent.SET_GOAL, ConversationAction.SET_GOAL,
                  proposed_goal="Spring AOP"),
            _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="请解释 AOP"),
        ]),
        diagnoser_model=ScriptedLLM([]),
        checkpointer=InMemorySaver(),
        store=store,
    )
    graph1.invoke(
        {"user_message": "Spring AOP"},
        config={"configurable": {"thread_id": "t1", "user_id": "u1"}},
    )

    # 第二次会话：新 thread、新图实例、空 planner 队列；同 user + 同 goal 应复用
    planner2 = ScriptedLLM([])  # 若错误地重新构建，invoke 会因队列耗尽而 assert
    graph2 = build_graph(
        planner_model=planner2,
        teacher_model=ScriptedLLM([
            _turn(ConversationIntent.SET_GOAL, ConversationAction.SET_GOAL,
                  proposed_goal="Spring AOP"),
            _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="再解释一次"),
        ]),
        diagnoser_model=ScriptedLLM([]),
        checkpointer=InMemorySaver(),
        store=store,
    )
    config2 = {"configurable": {"thread_id": "t2", "user_id": "u1"}}
    graph2.invoke({"user_message": "Spring AOP"}, config=config2)

    assert planner2.calls == []  # 复用：planner 未被再次调用
    state2 = graph2.get_state(config2).values
    assert state2["knowledge_model"]["points"][0]["id"] == "aop-concept"
    assert state2["goal"] == "Spring AOP"


def test_cross_session_goal_roundtrip_reads_back_mastery():
    """端到端：第一次会话 mastered 并写 delta，第二次同 user + 同 goal 读回 mastered。

    覆盖 Task ⑨ 验收标准：知识模型冻结 + point_id 稳定 + 读回闭环真正闭合。
    """
    store = InMemoryStore()

    # 第一次会话：完整 mastered 闭环
    graph1 = build_graph(
        planner_model=ScriptedLLM([_single_point_model()]),
        teacher_model=ScriptedLLM([
            _turn(ConversationIntent.SET_GOAL, ConversationAction.SET_GOAL,
                  proposed_goal="Spring AOP"),
            _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="请解释 AOP"),
            _turn(ConversationIntent.ANSWER, ConversationAction.DIAGNOSE),
            _turn(ConversationIntent.ANSWER, ConversationAction.RESPOND, reply="恭喜掌握 AOP"),
        ]),
        diagnoser_model=ScriptedLLM([
            _mastered_diagnosis(),
            _ConceptAssessmentStub(True, "概念解释正确"),
            _ScenarioAssessmentStub(True, "场景辨析正确"),
        ]),
        checkpointer=InMemorySaver(),
        store=store,
    )
    config1 = {"configurable": {"thread_id": "t1", "user_id": "u1"}}
    graph1.invoke({"user_message": "Spring AOP"}, config=config1)
    final1 = graph1.invoke(
        Command(resume="AOP 是面向切面编程，通过切面拦截方法调用……"),
        config=config1,
    )
    # 模拟 app 层持久化：把 graph 产出的 Delta 写入 Store
    for delta in final1["proficiency_deltas"]:
        append_proficiency_delta(store, "u1", ProficiencyEntry.model_validate(delta))

    # 第二次会话：同 user + 同 goal，空 planner（复用 km）
    graph2 = build_graph(
        planner_model=ScriptedLLM([]),
        teacher_model=ScriptedLLM([
            _turn(ConversationIntent.SET_GOAL, ConversationAction.SET_GOAL,
                  proposed_goal="Spring AOP"),
            _turn(ConversationIntent.ANSWER, ConversationAction.PROBE, reply="再问一次"),
        ]),
        diagnoser_model=ScriptedLLM([]),
        checkpointer=InMemorySaver(),
        store=store,
    )
    config2 = {"configurable": {"thread_id": "t2", "user_id": "u1"}}
    graph2.invoke({"user_message": "Spring AOP"}, config=config2)

    state2 = graph2.get_state(config2).values
    assert state2["knowledge_model"]["points"][0]["id"] == "aop-concept"
    assert state2["current_long_state"] == "mastered"  # 跨会话读回历史熟练度
