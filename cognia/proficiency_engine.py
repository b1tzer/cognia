"""Cognia 熟练度引擎（能力域 C）：BKT 纯数学融合。

BKT = Bayesian Knowledge Tracing（贝叶斯知识追踪）。把 AI 提交的五态观察样本
映射为二元观测（correct / incorrect），再用贝叶斯后验更新算出 P(learned) 连续值。

本模块只放「纯数学」，零 LLM 依赖、零 IO、纯函数：
- 不读 Store、不落库（IO 由 memory.py 与后续 compute_proficiency 负责）。
- 输入观察状态序列 + BKT 参数，输出后验概率。

分层（对齐宪法「决策点 vs 记忆点」）：
- 数值更新（P(learned) 后验）是「记忆点」，用确定性数学，永不被 AI 控制流篡改。
- 参数非法 / 缺失 → 回退默认值 / 钳制到 [0,1]，绝不抛异常中断融合流程。
"""

from dataclasses import dataclass

from cognia.schemas import BloomLevel, CognitiveState

# ---- BKT 四参数默认值 ----
# P(L0)：先验已掌握概率；P(T)：学习转移概率；P(G)：猜测概率；P(S)：失误概率。
DEFAULT_P_L0 = 0.40
DEFAULT_P_T = 0.10
DEFAULT_P_G = 0.20
DEFAULT_P_S = 0.10

# 难度 → 学习转移概率 P(T)：难度越高，每次交互的「学习转移」越少。
_P_T_BY_DIFFICULTY = {
    1: 0.20,
    2: 0.15,
    3: 0.10,
    4: 0.06,
    5: 0.03,
}

_EPS = 1e-12


def _clamp(value: float) -> float:
    """钳制到 [0, 1]，非法值（None / 越界）回退到 0.0。"""
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


@dataclass
class BKTParams:
    """BKT 四参数（构造时自动钳制到 [0,1]，非法值不抛异常）。"""

    p_l0: float = DEFAULT_P_L0
    p_t: float = DEFAULT_P_T
    p_g: float = DEFAULT_P_G
    p_s: float = DEFAULT_P_S

    def __post_init__(self) -> None:
        self.p_l0 = _clamp(self.p_l0)
        self.p_t = _clamp(self.p_t)
        self.p_g = _clamp(self.p_g)
        self.p_s = _clamp(self.p_s)


def params_for_difficulty(difficulty) -> BKTParams:
    """按知识点难度返回 BKT 参数（难度越高 P(T) 越低，其余参数用默认值）。

    非法 / 缺失难度 → 回退默认 P(T)，不抛异常。
    """
    try:
        d = int(difficulty)
    except (TypeError, ValueError):
        d = -1
    p_t = _P_T_BY_DIFFICULTY.get(d, DEFAULT_P_T)
    return BKTParams(p_t=p_t)


def state_to_binary(state) -> bool | None:
    """五态观察值 → BKT 二元观测。

    - mastered / partial → True（correct，答对）
    - misconception / unknown → False（incorrect，答错）
    - unassessed → None（跳过，不产生观测）

    传入非 CognitiveState 值时按 unassessed 处理（跳过），不抛异常。
    """
    if state == CognitiveState.MASTERED or state == CognitiveState.PARTIAL:
        return True
    if state == CognitiveState.MISCONCEPTION or state == CognitiveState.UNKNOWN:
        return False
    return None


def bkt_update(p_learned: float, is_correct: bool | None, params: BKTParams) -> float:
    """单次观察后更新 P(learned)（标准 BKT：先学习转移，再贝叶斯观察更新）。

    - is_correct 为 None 时不更新（跳过），原样返回。
    - 参数被钳制到 [0,1]；分母接近 0（模型无法从该观测推断）时原样返回，不除零。
    """
    if is_correct is None:
        return _clamp(p_learned)

    p = _clamp(p_learned)
    p_t = _clamp(params.p_t)
    p_g = _clamp(params.p_g)
    p_s = _clamp(params.p_s)

    # 学习转移：每次交互先有一次学习机会
    p = p + (1.0 - p) * p_t

    if is_correct:
        # P(L | correct) = P(L)(1-P(S)) / [P(L)(1-P(S)) + (1-P(L))P(G)]
        num = p * (1.0 - p_s)
        den = num + (1.0 - p) * p_g
    else:
        # P(L | incorrect) = P(L)P(S) / [P(L)P(S) + (1-P(L))(1-P(G))]
        num = p * p_s
        den = num + (1.0 - p) * (1.0 - p_g)

    if den < _EPS:
        # 分母趋近 0：观测不提供信息，保持当前后验
        return p
    return _clamp(num / den)


def bkt_infer(states, params: BKTParams | None = None) -> float:
    """融合一串五态观察，返回最终 P(learned) 后验。

    - 逐个跳过 unassessed（不产生观测）。
    - params 为 None 时用默认四参数。
    """
    p = BKTParams() if params is None else params
    current = p.p_l0
    for state in states:
        is_correct = state_to_binary(state)
        current = bkt_update(current, is_correct, p)
    return current


# ---- 离散化映射（能力域 C：连续值 → 五态）----

# 判 mastered 的连续值阈值（P(learned) 下限，实现阶段可调参）。
DEFAULT_MASTERY_THRESHOLD = 0.75


def verification_depth(bloom_level) -> int:
    """由认知层级派生「升到 mastered 所需的最少独立观察次数」。

    - remember / understand → 1 次（一次独立观察即可定级）。
    - apply / analyze / evaluate / create → 2 次（概念 + 场景两次独立观察）。

    非法 / 缺失 bloom_level → 保守取 2（宁可多验证一次，不降低定级门槛）。
    """
    if bloom_level == BloomLevel.REMEMBER or bloom_level == BloomLevel.UNDERSTAND:
        return 1
    return 2


def _effective_states(states) -> list:
    """过滤 unassessed，只保留产生观测的状态（保持原顺序）。"""
    return [s for s in states if state_to_binary(s) is not None]


def discretize(
    states,
    latent_value: float,
    verification_depth: int,
    mastery_threshold: float = DEFAULT_MASTERY_THRESHOLD,
) -> CognitiveState:
    """把连续值 P(learned) + 观察序列离散化为五态。

    规则（对齐 design §5.3）：
    1. 有效观察数为 0 → unassessed（未评估，与 unknown 区分）。
    2. `latent_value ≥ mastery_threshold` 且 `有效观察数 ≥ verification_depth`
       → mastered（阈值 + 最少观察次数双条件）。
    3. 未达 mastered：
       - 最近一次负向观察为 misconception → misconception
       - 最近一次负向观察为 unknown → unknown
       - 否则（存在正确性证据但未达阈值）→ partial

    `verification_depth` 由调用方经 `verification_depth(bloom_level)` 派生后传入。
    `mastery_threshold` 非法时钳制到 [0,1]。
    """
    effective = _effective_states(states)
    count = len(effective)
    if count == 0:
        return CognitiveState.UNASSESSED

    threshold = _clamp(mastery_threshold)
    if latent_value >= threshold and count >= verification_depth:
        return CognitiveState.MASTERED

    # 最近一次负向观察决定 misconception vs unknown（两者语义不同）。
    for state in reversed(effective):
        if state == CognitiveState.MISCONCEPTION:
            return CognitiveState.MISCONCEPTION
        if state == CognitiveState.UNKNOWN:
            return CognitiveState.UNKNOWN

    # 有正确性证据（存在 correct 观测）但未达 mastered 阈值 → partial。
    return CognitiveState.PARTIAL
