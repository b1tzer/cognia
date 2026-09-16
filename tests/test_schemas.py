"""cognia.schemas 数据模型单元测试。"""

import pytest

from cognia.schemas import (
    CognitiveState,
    Confidence,
    Diagnosis,
    Intervention,
    KnowledgeModel,
    KnowledgePoint,
    ProficiencyEntry,
    ValidationResult,
    VerificationState,
)


def test_cognitive_state_values():
    """五态枚举值正确。"""
    assert CognitiveState.UNASSESSED.value == "unassessed"
    assert CognitiveState.MASTERED.value == "mastered"
    assert CognitiveState.PARTIAL.value == "partial"
    assert CognitiveState.MISCONCEPTION.value == "misconception"
    assert CognitiveState.UNKNOWN.value == "unknown"


def test_confidence_values():
    """置信度三级枚举值正确。"""
    assert Confidence.HIGH.value == "high"
    assert Confidence.MEDIUM.value == "medium"
    assert Confidence.LOW.value == "low"


def test_validation_result_values():
    """验证结果三态枚举值正确。"""
    assert ValidationResult.UNASSESSED.value == "unassessed"
    assert ValidationResult.PASSED.value == "passed"
    assert ValidationResult.FAILED.value == "failed"


def test_proficiency_entry_fields():
    """ProficiencyEntry 字段齐全，update_type 固定为 delta，timestamp 默认带时区。"""
    entry = ProficiencyEntry(
        point_id="aop-proxy",
        from_state=CognitiveState.UNASSESSED,
        to_state=CognitiveState.PARTIAL,
        evidence=["用户说：AOP 就是切面，但说不清代理怎么实现"],
    )
    assert entry.from_state == CognitiveState.UNASSESSED
    assert entry.to_state == CognitiveState.PARTIAL
    assert entry.update_type == "delta"
    assert entry.evidence == ["用户说：AOP 就是切面，但说不清代理怎么实现"]
    assert entry.timestamp.tzinfo is not None  # 带时区（UTC）


def test_proficiency_entry_first_diagnosis_from_state_none():
    """首次诊断时 from_state 可为 None。"""
    entry = ProficiencyEntry(
        point_id="aop-proxy",
        to_state=CognitiveState.UNKNOWN,
    )
    assert entry.from_state is None
    assert entry.update_type == "delta"


def test_proficiency_entry_json_roundtrip():
    """ProficiencyEntry 的 timestamp（datetime）可 JSON 序列化 / 反序列化。"""
    entry = ProficiencyEntry(
        point_id="aop-proxy",
        from_state=CognitiveState.PARTIAL,
        to_state=CognitiveState.MASTERED,
        evidence=["用户说：装饰器是函数"],
    )
    data = entry.model_dump_json()
    restored = ProficiencyEntry.model_validate_json(data)
    assert restored.timestamp == entry.timestamp
    assert restored.to_state == CognitiveState.MASTERED


def test_json_roundtrip():
    """模型可 JSON 序列化 / 反序列化。"""
    diag = Diagnosis(
        point_id="aop-pointcut",
        state=CognitiveState.MISCONCEPTION,
        confidence=Confidence.HIGH,
        evidence=["用户认为 @Around 会修改原方法字节码"],
    )
    data = diag.model_dump_json()
    restored = Diagnosis.model_validate_json(data)
    assert restored == diag


def test_verification_state_defaults():
    """VerificationState 默认三态均为 unassessed。"""
    v = VerificationState()
    assert v.concept == ValidationResult.UNASSESSED
    assert v.scenario == ValidationResult.UNASSESSED
    assert v.current_step == "concept"


def test_knowledge_model():
    """KnowledgeModel 与 KnowledgePoint 可构造。"""
    point = KnowledgePoint(
        id="aop-proxy",
        name="AOP 代理机制",
        description="JDK 动态代理与 CGLIB 的区别",
        prerequisites=["aop-concept"],
    )
    model = KnowledgeModel(goal="Spring AOP", points=[point])
    assert model.goal == "Spring AOP"
    assert model.points[0].prerequisites == ["aop-concept"]


def test_intervention():
    """Intervention 可构造，干预类型为四选一。"""
    inter = Intervention(
        point_id="aop-proxy",
        intervention_type="correct",
        content="JDK 动态代理只能代理接口，CGLIB 通过继承实现",
    )
    assert inter.intervention_type == "correct"


def test_diagnosis_evidence_coerces_json_string():
    """evidence 被 LLM 误输出为 JSON 数组字符串时，归一化为 list[str]。"""
    diag = Diagnosis(
        point_id="aop-proxy",
        state=CognitiveState.UNKNOWN,
        confidence=Confidence.HIGH,
        evidence='["@Transactional 不是自动生效的"]',
    )
    assert diag.evidence == ["@Transactional 不是自动生效的"]


def test_diagnosis_evidence_coerces_plain_string():
    """evidence 为单个纯字符串时，归一化为单元素 list。"""
    diag = Diagnosis(
        point_id="aop-proxy",
        state=CognitiveState.UNKNOWN,
        confidence=Confidence.HIGH,
        evidence="用户说没听过",
    )
    assert diag.evidence == ["用户说没听过"]


def test_diagnosis_evidence_list_unchanged():
    """evidence 为正常 list 时保持原样。"""
    diag = Diagnosis(
        point_id="aop-proxy",
        state=CognitiveState.PARTIAL,
        confidence=Confidence.HIGH,
        evidence=["能说大意", "细节模糊"],
    )
    assert diag.evidence == ["能说大意", "细节模糊"]
