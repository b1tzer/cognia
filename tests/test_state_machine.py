"""cognia.state_machine 状态机单元测试。

穷举 5×5=25 种组合，与 state-machine.md §1 总览表逐一比对。
"""

import pytest

from cognia.schemas import CognitiveState, ValidationResult, VerificationState
from cognia.state_machine import can_transition, is_mastered_migration_allowed

# 期望允许的迁移集合（与 state-machine.md §1 总览表的 ✅ 一一对应）
EXPECTED_ALLOWED = {
    (CognitiveState.UNASSESSED, CognitiveState.UNKNOWN),
    (CognitiveState.UNASSESSED, CognitiveState.PARTIAL),
    (CognitiveState.UNASSESSED, CognitiveState.MISCONCEPTION),
    (CognitiveState.UNASSESSED, CognitiveState.MASTERED),
    (CognitiveState.UNKNOWN, CognitiveState.PARTIAL),
    (CognitiveState.UNKNOWN, CognitiveState.MASTERED),
    (CognitiveState.PARTIAL, CognitiveState.MISCONCEPTION),
    (CognitiveState.PARTIAL, CognitiveState.MASTERED),
    (CognitiveState.MISCONCEPTION, CognitiveState.PARTIAL),
    (CognitiveState.MISCONCEPTION, CognitiveState.MASTERED),
    (CognitiveState.MASTERED, CognitiveState.UNKNOWN),
    (CognitiveState.MASTERED, CognitiveState.PARTIAL),
    (CognitiveState.MASTERED, CognitiveState.MISCONCEPTION),
}

# ---- 文档规格锚点（把测试锚死在 state-machine.md，而非两份代码互证）----

# state-machine.md §1 总览表的 ✅ 允许项总数（独立于实现，钉死文档规格）
EXPECTED_ALLOWED_COUNT = 13

# state-machine.md §3「禁止转换」表明确列出的 4 类禁止项
FORBIDDEN_TRANSITIONS = {
    (CognitiveState.UNKNOWN, CognitiveState.MISCONCEPTION),
    (CognitiveState.PARTIAL, CognitiveState.UNKNOWN),
    (CognitiveState.MISCONCEPTION, CognitiveState.UNKNOWN),
    (CognitiveState.MASTERED, CognitiveState.UNASSESSED),
}

ALL_STATES = list(CognitiveState)


def test_exhaustive_transition_matrix():
    """穷举 25 种组合，can_transition 与总览表逐一比对。"""
    for from_state in ALL_STATES:
        for to_state in ALL_STATES:
            expected = (from_state, to_state) in EXPECTED_ALLOWED
            assert can_transition(from_state, to_state) == expected, (
                f"{from_state.value} → {to_state.value} 期望 {expected}"
            )


def test_allowed_transitions_count_matches_spec():
    """允许项基数锚定文档规格（13 个 ✅），防止两份表同时漏抄某条而互证通过。"""
    assert len(EXPECTED_ALLOWED) == EXPECTED_ALLOWED_COUNT


def test_forbidden_transitions_count_matches_spec():
    """禁止项集合必须与 state-machine.md §3 的 4 类一一对应。"""
    assert len(FORBIDDEN_TRANSITIONS) == 4


@pytest.mark.parametrize("from_state, to_state", sorted(FORBIDDEN_TRANSITIONS))
def test_forbidden_transitions(from_state, to_state):
    """锁死 state-machine.md §3 明确列出的禁止转换项（锚定文档，非代码互证）。"""
    assert can_transition(from_state, to_state) is False


def test_no_state_transitions_back_to_unassessed():
    """unassessed 是唯一初始态，任何非初始态都不可回退到 unassessed（§1 总览表）。"""
    for from_state in ALL_STATES:
        if from_state is CognitiveState.UNASSESSED:
            continue
        assert can_transition(from_state, CognitiveState.UNASSESSED) is False


@pytest.mark.parametrize("from_state", ALL_STATES)
def test_self_transition_is_not_a_transition(from_state):
    """自我迁移不算转换（总览表对角线的「—」）。"""
    assert can_transition(from_state, from_state) is False


def test_mastered_migration_allowed_when_both_pass():
    """双重验证全过 → 允许迁移到 mastered。"""
    v = VerificationState(
        concept=ValidationResult.PASSED,
        scenario=ValidationResult.PASSED,
        current_step="done",
    )
    assert is_mastered_migration_allowed(v) is True


def test_mastered_migration_blocked_when_concept_failed():
    """概念验证失败 → 禁止迁移到 mastered。"""
    v = VerificationState(
        concept=ValidationResult.FAILED,
        scenario=ValidationResult.PASSED,
    )
    assert is_mastered_migration_allowed(v) is False


def test_mastered_migration_blocked_when_scenario_unassessed():
    """场景验证未评估 → 禁止迁移到 mastered。"""
    v = VerificationState(
        concept=ValidationResult.PASSED,
        scenario=ValidationResult.UNASSESSED,
    )
    assert is_mastered_migration_allowed(v) is False


def test_mastered_migration_blocked_when_all_unassessed():
    """默认全 unassessed → 禁止迁移到 mastered。"""
    v = VerificationState()
    assert is_mastered_migration_allowed(v) is False
