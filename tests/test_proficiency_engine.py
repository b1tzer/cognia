"""cognia.proficiency_engine 熟练度引擎（BKT）单元测试。

纯数学、零 LLM 依赖、零 IO，用内存直接驱动，验证：
1. 五态 → 二元观测映射
2. 后验更新方向（答对上升 / 答错下降）
3. 参数钳制与非法回退（不抛异常）
4. 难度查表（难度越高 P(T) 越低）
5. 序列融合与除零保护
6. 离散化（verification_depth 派生 + 连续值 → 五态）
7. compute_proficiency（观察历史 + 属性 → 权威 Proficiency）
"""

from datetime import datetime, timezone

from cognia.proficiency_engine import (
    DEFAULT_P_L0,
    DEFAULT_P_T,
    DEFAULT_MASTERY_THRESHOLD,
    BKTParams,
    bkt_infer,
    bkt_update,
    compute_proficiency,
    discretize,
    params_for_difficulty,
    state_to_binary,
    verification_depth,
)
from cognia.schemas import (
    BloomLevel,
    CognitiveState,
    Confidence,
    Observation,
    PointAttributes,
    PointType,
    Proficiency,
)


def test_state_to_binary_correct():
    """mastered / partial → True（correct）。"""
    assert state_to_binary(CognitiveState.MASTERED) is True
    assert state_to_binary(CognitiveState.PARTIAL) is True


def test_state_to_binary_incorrect():
    """misconception / unknown → False（incorrect）。"""
    assert state_to_binary(CognitiveState.MISCONCEPTION) is False
    assert state_to_binary(CognitiveState.UNKNOWN) is False


def test_state_to_binary_unassessed_skipped():
    """unassessed → None（跳过，不产生观测）。"""
    assert state_to_binary(CognitiveState.UNASSESSED) is None


def test_state_to_binary_invalid_skipped():
    """非法值 → None（按 unassessed 跳过，不抛异常）。"""
    assert state_to_binary(None) is None
    assert state_to_binary("whatever") is None


def test_bkt_update_correct_raises():
    """答对后 P(learned) 上升。"""
    params = BKTParams()
    after = bkt_update(DEFAULT_P_L0, True, params)
    assert after > DEFAULT_P_L0


def test_bkt_update_incorrect_drops():
    """答错后 P(learned) 下降。"""
    params = BKTParams()
    after = bkt_update(DEFAULT_P_L0, False, params)
    assert after < DEFAULT_P_L0


def test_bkt_update_none_returns_same():
    """is_correct 为 None 时不更新，原样返回（钳制后）。"""
    params = BKTParams()
    assert bkt_update(0.5, None, params) == 0.5


def test_bkt_update_clamps_p_learned():
    """p_learned 越界被钳制到 [0,1]。"""
    params = BKTParams()
    assert 0.0 <= bkt_update(5.0, True, params) <= 1.0
    assert 0.0 <= bkt_update(-3.0, False, params) <= 1.0


def test_bkt_params_clamps_fields():
    """BKTParams 构造时把非法参数钳制到 [0,1]，不抛异常。"""
    p = BKTParams(p_l0=2.0, p_t=-0.5, p_g=1.5, p_s=-1.0)
    assert p.p_l0 == 1.0
    assert p.p_t == 0.0
    assert p.p_g == 1.0
    assert p.p_s == 0.0


def test_bkt_params_keeps_valid_fields():
    """合法参数保持不变。"""
    p = BKTParams(p_l0=0.3, p_t=0.15, p_g=0.2, p_s=0.1)
    assert p.p_l0 == 0.3
    assert p.p_t == 0.15
    assert p.p_g == 0.2
    assert p.p_s == 0.1


def test_params_for_difficulty_decreasing_p_t():
    """难度越高 P(T) 越低。"""
    assert params_for_difficulty(1).p_t > params_for_difficulty(3).p_t
    assert params_for_difficulty(3).p_t > params_for_difficulty(5).p_t


def test_params_for_difficulty_exact_values():
    """查表值精确：difficulty=1 → 0.20，difficulty=5 → 0.03。"""
    assert params_for_difficulty(1).p_t == 0.20
    assert params_for_difficulty(5).p_t == 0.03


def test_params_for_difficulty_invalid_falls_back():
    """非法 / 缺失难度回退默认 P(T)。"""
    assert params_for_difficulty(99).p_t == DEFAULT_P_T
    assert params_for_difficulty(None).p_t == DEFAULT_P_T
    assert params_for_difficulty("abc").p_t == DEFAULT_P_T


def test_bkt_infer_empty_returns_prior():
    """无观测 → 返回先验 P(L0)。"""
    assert bkt_infer([]) == DEFAULT_P_L0


def test_bkt_infer_skips_unassessed():
    """unassessed 被跳过，不影响结果。"""
    assert bkt_infer([CognitiveState.MASTERED]) == bkt_infer(
        [CognitiveState.UNASSESSED, CognitiveState.MASTERED]
    )


def test_bkt_infer_more_correct_higher():
    """答对越多 P(learned) 越高。"""
    one = bkt_infer([CognitiveState.MASTERED])
    two = bkt_infer([CognitiveState.MASTERED, CognitiveState.MASTERED])
    assert two > one


def test_bkt_infer_incorrect_drops_below_prior():
    """答错一次后 P(learned) 低于先验。"""
    assert bkt_infer([CognitiveState.UNKNOWN]) < DEFAULT_P_L0


def test_bkt_update_division_by_zero_safe():
    """极端参数（correct 分母为 0）不除零，返回当前后验。"""
    params = BKTParams(p_l0=0.5, p_t=0.0, p_g=0.0, p_s=1.0)
    assert bkt_update(0.5, True, params) == 0.5


def test_bkt_update_incorrect_division_by_zero_safe():
    """极端参数（incorrect 分母为 0）不除零，返回当前后验。"""
    params = BKTParams(p_l0=0.5, p_t=0.0, p_g=1.0, p_s=0.0)
    assert bkt_update(0.5, False, params) == 0.5


# ---- 离散化映射（verification_depth 派生 + 连续值 → 五态）----

def test_verification_depth_remember_understand_one():
    """remember / understand 派生 1 次验证深度。"""
    assert verification_depth(BloomLevel.REMEMBER) == 1
    assert verification_depth(BloomLevel.UNDERSTAND) == 1


def test_verification_depth_apply_above_two():
    """apply / analyze / evaluate / create 派生 2 次验证深度。"""
    assert verification_depth(BloomLevel.APPLY) == 2
    assert verification_depth(BloomLevel.ANALYZE) == 2
    assert verification_depth(BloomLevel.EVALUATE) == 2
    assert verification_depth(BloomLevel.CREATE) == 2


def test_verification_depth_invalid_falls_back_two():
    """非法 / 缺失 bloom_level 保守取 2（不降低定级门槛）。"""
    assert verification_depth(None) == 2
    assert verification_depth("whatever") == 2


def test_discretize_no_observations_unassessed():
    """无有效观察 → unassessed（与 unknown 区分）。"""
    assert discretize([], 0.0, 1) == CognitiveState.UNASSESSED
    assert discretize([CognitiveState.UNASSESSED], 0.0, 1) == CognitiveState.UNASSESSED


def test_discretize_mastered_threshold_and_depth():
    """latent ≥ 阈值 且 观察数 ≥ depth → mastered。"""
    assert discretize([CognitiveState.MASTERED], 0.9, 1) == CognitiveState.MASTERED


def test_discretize_mastered_requires_depth():
    """latent ≥ 阈值 但观察数不足 depth → 不定 mastered（apply 需 2 次）。"""
    # 只有 1 次 correct 观察，depth=2，即便 latent 高也不 mastered
    result = discretize([CognitiveState.MASTERED], 0.9, 2)
    assert result != CognitiveState.MASTERED
    assert result == CognitiveState.PARTIAL


def test_discretize_below_threshold_partial():
    """latent 未达阈值 → partial（存在正确性证据）。"""
    assert discretize([CognitiveState.PARTIAL], 0.5, 1) == CognitiveState.PARTIAL


def test_discretize_recent_misconception_wins():
    """最近一次负向观察为 misconception → misconception。"""
    assert discretize(
        [CognitiveState.MASTERED, CognitiveState.MISCONCEPTION], 0.3, 1
    ) == CognitiveState.MISCONCEPTION


def test_discretize_recent_unknown_wins():
    """最近一次负向观察为 unknown → unknown。"""
    assert discretize(
        [CognitiveState.PARTIAL, CognitiveState.UNKNOWN], 0.2, 1
    ) == CognitiveState.UNKNOWN


def test_discretize_negative_overrides_partial():
    """有正确证据但最近为负向 → 负向类型优先于 partial。"""
    assert discretize(
        [CognitiveState.MASTERED, CognitiveState.UNKNOWN], 0.6, 1
    ) == CognitiveState.UNKNOWN


def test_discretize_threshold_clamped():
    """mastery_threshold 非法值被钳制到 [0,1]，不抛异常。"""
    # 阈值钳制为 1.0 时，latent 0.9 无法 mastered
    assert discretize([CognitiveState.MASTERED], 0.9, 1, mastery_threshold=5.0) == CognitiveState.PARTIAL
    # 阈值钳制为 0.0 时，任意 latent 都可能 mastered（只要观察数够）
    assert discretize([CognitiveState.MASTERED], 0.0, 1, mastery_threshold=-1.0) == CognitiveState.MASTERED


# ---- compute_proficiency（观察历史 + 属性 → 权威 Proficiency）----

def _obs(point_id, state, day, confidence=Confidence.HIGH):
    """构造带确定 timestamp 的 Observation。"""
    return Observation(
        point_id=point_id,
        observed_state=state,
        confidence=confidence,
        evidence=[f"证据-{day}"],
        timestamp=datetime(2026, 9, day, tzinfo=timezone.utc),
    )


def _attrs(bloom=BloomLevel.UNDERSTAND, difficulty=2):
    """构造 PointAttributes（默认 understand + difficulty=2）。"""
    return PointAttributes(
        type=PointType.CONCEPT,
        difficulty=difficulty,
        importance=3,
        bloom_level=bloom,
    )


def test_compute_proficiency_empty_unassessed():
    """空观察 → unassessed，point_id 空，observation_count=0，latent=先验。"""
    p = compute_proficiency([], _attrs())
    assert p.mapped_state == CognitiveState.UNASSESSED
    assert p.point_id == ""
    assert p.observation_count == 0
    assert p.latent_value == DEFAULT_P_L0


def test_compute_proficiency_single_mastered_understand():
    """单次 mastered + understand（depth=1）→ mastered。"""
    p = compute_proficiency(
        [_obs("p1", CognitiveState.MASTERED, 1)],
        _attrs(bloom=BloomLevel.UNDERSTAND),
    )
    assert p.mapped_state == CognitiveState.MASTERED
    assert p.observation_count == 1


def test_compute_proficiency_single_mastered_apply_requires_two():
    """单次 mastered + apply（depth=2）→ 观察数不足，不定 mastered（partial）。"""
    p = compute_proficiency(
        [_obs("p1", CognitiveState.MASTERED, 1)],
        _attrs(bloom=BloomLevel.APPLY),
    )
    assert p.mapped_state != CognitiveState.MASTERED
    assert p.mapped_state == CognitiveState.PARTIAL
    assert p.observation_count == 1


def test_compute_proficiency_accepts_dicts():
    """observations 可传 dict（query_observations 的 json 返回值）。"""
    raw = _obs("p1", CognitiveState.MASTERED, 1).model_dump(mode="json")
    p = compute_proficiency([raw], _attrs(bloom=BloomLevel.UNDERSTAND))
    assert p.point_id == "p1"
    assert p.observation_count == 1


def test_compute_proficiency_skips_invalid_items():
    """非法观察项被跳过，不影响整体计算。"""
    p = compute_proficiency(
        [None, "not-an-obs", _obs("p1", CognitiveState.MASTERED, 1)],
        _attrs(bloom=BloomLevel.UNDERSTAND),
    )
    assert p.point_id == "p1"
    assert p.observation_count == 1


def test_compute_proficiency_point_id_and_last_updated():
    """point_id 取第一个观察，last_updated 取最后一个观察的 timestamp。"""
    o1 = _obs("p1", CognitiveState.PARTIAL, 1)
    o2 = _obs("p1", CognitiveState.MASTERED, 2)
    p = compute_proficiency([o1, o2], _attrs(bloom=BloomLevel.UNDERSTAND))
    assert p.point_id == "p1"
    assert p.last_updated == o2.timestamp


def test_compute_proficiency_misconception():
    """最近一次负向观察为 misconception → misconception。"""
    p = compute_proficiency(
        [_obs("p1", CognitiveState.MASTERED, 1), _obs("p1", CognitiveState.MISCONCEPTION, 2)],
        _attrs(bloom=BloomLevel.UNDERSTAND),
    )
    assert p.mapped_state == CognitiveState.MISCONCEPTION


def test_compute_proficiency_fields_complete():
    """Proficiency 字段齐全且数值范围合法。"""
    p = compute_proficiency(
        [_obs("p1", CognitiveState.MASTERED, 1)],
        _attrs(bloom=BloomLevel.UNDERSTAND),
    )
    assert isinstance(p, Proficiency)
    assert 0.0 <= p.latent_value <= 1.0
    assert 0.0 <= p.uncertainty <= 1.0
    assert p.source_algorithm == "bkt"
    assert p.last_updated.tzinfo is not None
