"""cognia.tools 认知模型工具层单元测试。

全部使用 ScriptedLLM 假模型 + InMemoryStore，离线、快速、不依赖 API。

重点验证：
1. 读工具：read_learner_state 正确读五态。
2. 写工具 propose_diagnosis（诊断=提交观察，系统 BKT 定级）：
   - 提交观察样本、不直接写 Delta；
   - 返回系统权威状态（authoritative_state）；
   - 不再走双重验证；
   - store 缺失安全降级。
3. record_observation / query_proficiency skill（提交观察值 + 查询权威状态）。
4. 教学工具 generate_probe / explain 生成文本。
5. build_learning_goal 的 load-or-build 复用。
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

    def with_structured_output(self, schema, **kwargs):
        # 对齐真实 LangChain 模型签名：structured_output 会传 tool_choice="auto"
        # 等 kwargs，mock 需接受但忽略。
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        assert self._responses, "ScriptedLLM 响应队列耗尽"
        return self._responses.pop(0)


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
    """读工具返回系统权威五态（Observation → BKT → mapped_state；未评估为 unassessed）。"""
    from cognia.memory import record_observation
    from cognia.schemas import Observation

    store = InMemoryStore()
    record_observation(store, "u1", Observation(
        point_id="aop-concept",
        observed_state=CognitiveState.PARTIAL,
        confidence=Confidence.HIGH,
        evidence=["e"],
    ))
    tools = build_cognia_tools(store=store)

    assert tools["read_learner_state"].invoke(
        {"point_id": "aop-concept"}, config=_cfg()
    ) == '{"state": "partial"}'
    assert tools["read_learner_state"].invoke(
        {"point_id": "unknown-point"}, config=_cfg()
    ) == '{"state": "unassessed"}'


# ---- 写工具：propose_diagnosis（诊断=提交观察，系统 BKT 定级）----

def test_propose_diagnosis_submits_observation_and_returns_authoritative():
    """propose_diagnosis 提交观察样本并返回系统权威状态（不直接写 Delta）。"""
    diagnoser = ScriptedLLM([_partial_diagnosis()])
    store = InMemoryStore()
    tools = build_cognia_tools(diagnoser=diagnoser, store=store)

    result = json.loads(tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "AOP 就是切面，但代理机制说不清",
        "current_state": "unassessed",
    }, config=_cfg()))

    assert result["recorded"] is True
    assert result["diagnosed_state"] == "partial"
    assert "authoritative_state" in result

    from cognia.memory import query_observations
    # 关键：只追加观察样本，权威状态由系统 BKT 融合算出
    assert len(query_observations(store, "u1", "aop-concept")) == 1


def test_propose_diagnosis_mastered_observation_authoritative_partial():
    """单次 mastered 观察 + 无属性（depth 保守 2）→ 权威状态 partial（不 mastered）。"""
    diagnoser = ScriptedLLM([_mastered_diagnosis()])
    store = InMemoryStore()
    tools = build_cognia_tools(diagnoser=diagnoser, store=store)

    result = json.loads(tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "AOP 通过切面拦截方法调用",
        "current_state": "partial",
    }, config=_cfg()))

    assert result["recorded"] is True
    assert result["diagnosed_state"] == "mastered"
    # mastered 双条件：latent≥阈值 且 观察数≥depth；无 attributes 时 depth=2，
    # 单次观察不足以定 mastered
    assert result["authoritative_state"] == "partial"
    assert result["observation_count"] == 1


def test_propose_diagnosis_no_double_verification():
    """改造后不再走双重验证：一次诊断只调用一次 diagnoser（无 verifier）。"""
    diagnoser = ScriptedLLM([_mastered_diagnosis()])
    store = InMemoryStore()
    tools = build_cognia_tools(diagnoser=diagnoser, store=store)

    tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "AOP 通过切面拦截方法调用",
        "current_state": "unassessed",
    }, config=_cfg())

    # 只调用一次 diagnoser（run_diagnosis），没有第二次 verifier 调用
    assert len(diagnoser.calls) == 1


def test_propose_diagnosis_store_none_degrades():
    """store 为 None 时安全降级：不落库，返回 unassessed 权威状态。"""
    diagnoser = ScriptedLLM([_partial_diagnosis()])
    tools = build_cognia_tools(diagnoser=diagnoser, store=None)

    result = json.loads(tools["propose_diagnosis"].invoke({
        "point_id": "aop-concept",
        "point_name": "AOP 概念", "point_description": "面向切面编程",
        "question": "什么是 AOP？", "user_answer": "AOP 就是切面",
        "current_state": "unassessed",
    }, config=_cfg()))

    assert result["recorded"] is False
    assert result["authoritative_state"] == "unassessed"


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

def test_build_learning_goal_builds_then_reuses(monkeypatch):
    """首次构建知识模型并冻结，第二次复用（planner 不再被调用）。

    point_id 会被重写为跨对话全局稳定 id（concept: 前缀），而非 LLM 裸 id。
    """
    # mock embedding：单测不触发真实模型下载，merge 退化为归一化 name 精确匹配
    monkeypatch.setattr("cognia.embedding.embed_text", lambda t: None)
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
    assert first["first_point_id"].startswith("concept:")
    first_id = first["first_point_id"]

    # 第二次：复用，planner 不再被调用（队列已空，若重建会 assert 耗尽）
    second = json.loads(tools["build_learning_goal"].invoke({"goal": "Spring AOP"}, config=_cfg()))
    assert second["action"] == "复用"
    assert second["first_point_id"] == first_id


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

    from cognia.memory import query_observations
    obs = query_observations(store, "u1", "aop-concept")
    assert len(obs) == 1
    assert obs[0]["observed_state"] == "partial"


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
