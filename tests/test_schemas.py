"""cognia.schemas 数据模型单元测试。"""

import pytest

from cognia.schemas import (
    BloomLevel,
    CognitiveState,
    Confidence,
    Diagnosis,
    KnowledgeModel,
    KnowledgePoint,
    Observation,
    PointAttributes,
    PointType,
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


def test_point_type_values():
    """PointType 枚举值正确。"""
    assert PointType.CONCEPT.value == "concept"
    assert PointType.FACT.value == "fact"
    assert PointType.SKILL.value == "skill"
    assert PointType.PRINCIPLE.value == "principle"
    assert PointType.PROCESS.value == "process"


def test_bloom_level_values():
    """BloomLevel 枚举值正确（六层）。"""
    assert BloomLevel.REMEMBER.value == "remember"
    assert BloomLevel.UNDERSTAND.value == "understand"
    assert BloomLevel.APPLY.value == "apply"
    assert BloomLevel.ANALYZE.value == "analyze"
    assert BloomLevel.EVALUATE.value == "evaluate"
    assert BloomLevel.CREATE.value == "create"


def test_point_attributes_fields():
    """PointAttributes 字段齐全，type/difficulty/importance/bloom_level 均可构造。"""
    attrs = PointAttributes(
        type=PointType.CONCEPT,
        difficulty=3,
        importance=4,
        bloom_level=BloomLevel.UNDERSTAND,
    )
    assert attrs.type == PointType.CONCEPT
    assert attrs.difficulty == 3
    assert attrs.importance == 4
    assert attrs.bloom_level == BloomLevel.UNDERSTAND


def test_point_attributes_difficulty_bounds():
    """difficulty / importance 越界（<1 或 >5）应被 Pydantic 拒绝。"""
    with pytest.raises(Exception):
        PointAttributes(
            type=PointType.SKILL,
            difficulty=0,
            importance=3,
            bloom_level=BloomLevel.APPLY,
        )
    with pytest.raises(Exception):
        PointAttributes(
            type=PointType.SKILL,
            difficulty=3,
            importance=6,
            bloom_level=BloomLevel.APPLY,
        )


def test_knowledge_point_with_attributes():
    """KnowledgePoint 可携带 attributes，JSON 往返保持。"""
    point = KnowledgePoint(
        id="kafka-what",
        name="Kafka 是什么",
        description="Kafka 是分布式流处理平台",
        prerequisites=[],
        attributes=PointAttributes(
            type=PointType.CONCEPT,
            difficulty=2,
            importance=5,
            bloom_level=BloomLevel.UNDERSTAND,
        ),
    )
    assert point.attributes.type == PointType.CONCEPT
    assert point.attributes.bloom_level == BloomLevel.UNDERSTAND

    restored = KnowledgePoint.model_validate_json(point.model_dump_json())
    assert restored.attributes == point.attributes


def test_knowledge_point_without_attributes_defaults_none():
    """KnowledgePoint 不传 attributes 时默认为 None（不破坏既有构造）。"""
    point = KnowledgePoint(
        id="aop-proxy",
        name="AOP 代理机制",
        description="JDK 动态代理与 CGLIB 的区别",
    )
    assert point.attributes is None


def test_observation_fields():
    """Observation 字段齐全，observer 默认 ai，timestamp 带时区。"""
    obs = Observation(
        point_id="kafka-what",
        observed_state=CognitiveState.PARTIAL,
        confidence=Confidence.HIGH,
        evidence=["用户说 Kafka 是高吞吐的消息队列，但说不清分区机制"],
    )
    assert obs.observed_state == CognitiveState.PARTIAL
    assert obs.confidence == Confidence.HIGH
    assert obs.evidence == ["用户说 Kafka 是高吞吐的消息队列，但说不清分区机制"]
    assert obs.observer == "ai"
    assert obs.timestamp.tzinfo is not None


def test_observation_json_roundtrip():
    """Observation 可 JSON 序列化 / 反序列化。"""
    obs = Observation(
        point_id="mq-concept",
        observed_state=CognitiveState.MASTERED,
        confidence=Confidence.HIGH,
        evidence=["用户说消息队列能解耦生产者和消费者"],
    )
    restored = Observation.model_validate_json(obs.model_dump_json())
    assert restored == obs


def test_observation_evidence_coerces_json_string():
    """Observation 的 evidence 被 LLM 误输出为 JSON 数组字符串时，归一化为 list[str]。"""
    obs = Observation(
        point_id="mq-concept",
        observed_state=CognitiveState.PARTIAL,
        confidence=Confidence.MEDIUM,
        evidence='["用户说消息队列就是异步"]',
    )
    assert obs.evidence == ["用户说消息队列就是异步"]
