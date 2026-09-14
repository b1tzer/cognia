"""Cognia 五态认知状态转移矩阵（纯函数）。

与 state-machine.md 一一对应，是 Cognia 核心状态机的可执行落地。

本模块只做「纯逻辑」，不含任何 LLM 调用、IO 或 LangGraph 依赖。
置信度门槛（高/中/低 → 迁移/冻结/探测）属于图逻辑层（任务⑤），不在此模块实现。

⚠️ 迁移到 mastered 必须同时满足三层闸门（任务⑤ 图逻辑层落地时的硬约束，
   缺一即构成 state-machine.md §5 规则 1 禁止的「直写旁路」）：

    1. can_transition(from_state, MASTERED)          # 拓扑合法（本模块）
    2. is_mastered_migration_allowed(verification)    # 双重验证全过（本模块）
    3. 置信度 == HIGH                                 # 置信度门槛（图逻辑层，本模块刻意不含）

    can_transition(UNASSESSED, MASTERED) == True 仅表示「拓扑上允许」，
    绝不等于「能进 mastered」；只查闸门 1 而跳过闸门 2/3，就是把单次
    diagnose 的 mastered 候选直写进长期状态，属违规。
"""

from cognia.schemas import CognitiveState, ValidationResult, VerificationState

# 允许的状态迁移集合（硬编码，来自 state-machine.md §1 总览表）
# 每个元素为 (from_state, to_state)
_ALLOWED_TRANSITIONS: frozenset[tuple[CognitiveState, CognitiveState]] = frozenset({
    # unassessed → 各态（首次诊断，四选一）
    (CognitiveState.UNASSESSED, CognitiveState.UNKNOWN),
    (CognitiveState.UNASSESSED, CognitiveState.PARTIAL),
    (CognitiveState.UNASSESSED, CognitiveState.MISCONCEPTION),
    (CognitiveState.UNASSESSED, CognitiveState.MASTERED),
    # unknown → 教学后
    (CognitiveState.UNKNOWN, CognitiveState.PARTIAL),
    (CognitiveState.UNKNOWN, CognitiveState.MASTERED),
    # partial → 深入探测 / 补全后
    (CognitiveState.PARTIAL, CognitiveState.MISCONCEPTION),
    (CognitiveState.PARTIAL, CognitiveState.MASTERED),
    # misconception → 纠错后
    (CognitiveState.MISCONCEPTION, CognitiveState.PARTIAL),
    (CognitiveState.MISCONCEPTION, CognitiveState.MASTERED),
    # mastered → 降级
    (CognitiveState.MASTERED, CognitiveState.UNKNOWN),
    (CognitiveState.MASTERED, CognitiveState.PARTIAL),
    (CognitiveState.MASTERED, CognitiveState.MISCONCEPTION),
})


def can_transition(from_state: CognitiveState, to_state: CognitiveState) -> bool:
    """判断 from_state → to_state 是否为状态机允许的迁移。

    与 state-machine.md §1 总览表一致：
    - ✅ 允许（需证据）
    - ❌ 禁止（含自我迁移 X→X、一切到 unassessed、以及 unknown→misconception、
      partial→unknown、misconception→unknown 等）
    """
    return (from_state, to_state) in _ALLOWED_TRANSITIONS


def is_mastered_migration_allowed(verification: VerificationState) -> bool:
    """判断是否允许迁移到 mastered（双重验证，三层闸门之第二层）。

    要求「概念解释」+「场景 / 反例辨析」两类验证都 PASSED
    （state-machine.md §1 关键前提、plan §3.5）。

    仅此函数返回 True 仍不足以迁移 mastered，还必须同时满足
    can_transition(from, MASTERED)（拓扑）与高置信度（图逻辑层）。
    """
    return (
        verification.concept == ValidationResult.PASSED
        and verification.scenario == ValidationResult.PASSED
    )
