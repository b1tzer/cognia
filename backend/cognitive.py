"""认知诊断引擎 + 贝叶斯知识追踪（BKT）学生模型。

职责：
1. diagnose：把用户的一段表达（对概念的理解陈述）分类为四状态
   理解 / 半理解 / 错误 / 信息不足，并映射到具体概念。
2. bayes_update：用贝叶斯规则更新每个概念的掌握概率。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config
from llm import chat_json
from schemas import Concept, DiagnosticResult

# ---------------------------------------------------------------------------
# LLM 诊断提示词
# ---------------------------------------------------------------------------
_DIAG_SYSTEM = """你是一名严谨的认知诊断专家。给你一个学习目标、一张概念列表、以及学习者
对某个概念的当前理解陈述，你要判断学习者的认知状态。

认知状态四分类：
- understood：理解正确、准确、能说清机制/因果，无明显错误
- partial：半理解，方向对但有遗漏、模糊、不完整
- misconceived：存在明确的概念错误或误解
- insufficient：信息不足，表达太空泛或干脆承认不知道，无法判断

判别原则（务必遵守，优先级最高）：
- 否定性证据优先：若表达中出现"不知道/不清楚/不会/没学过/忘了/不确定"等明确否定或承认不知道的信号，一律判为 insufficient，即使其他部分看似在谈相关概念。
- 判断「misconceived」必须基于「概念本身的语义错误」，不能仅因为表述口语化、不精确、或出现"其实/本质上"等措辞就误判为误解；若语义本质正确、只是措辞不够严谨，应归入 partial 或 understood。
- 判断「understood」应聚焦「最终结论的语义正确性」，不被表达过程中的犹豫、自我否定、或"说不清楚"等语气信号误导；只要最终结论正确且能说出关键机制，就应判为 understood。
- 追问兜底：若证据不足以支撑高置信判断，应降低 confidence（< 0.7），宁可判为 partial 并触发追问，也不要强判 understood 或 misconceived。

同时你要：
- concept_ids：这段陈述主要涉及哪些概念 id（从给定列表中选择）
- evidence：用一句话说明判断依据（引述学习者的原话要点）
- misconception：若存在误解，指出具体错在哪
- missing：指出还缺失哪些关键理解

只输出 JSON：
{"state":"partial","confidence":0.7,"concept_ids":["..."],"evidence":"...","misconception":"...","missing":["..."]}
"""

# ---------------------------------------------------------------------------
# 诊断经验库（「诊断自我迭代」的持久化载体）
#
# 记录从历史诊断错误中总结出的判别原则，注入诊断 prompt 以持续修正。
# 这是北极星指标「诊断准确度随对话自我迭代」的落地机制。
# ---------------------------------------------------------------------------
_RULES_PATH = Path(__file__).resolve().parent / "eval" / "diagnosis_rules.json"


def load_diagnosis_rules() -> list[str]:
    """加载诊断经验规则。文件缺失或损坏时安全降级为空列表。"""
    try:
        data = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
        rules = data.get("rules", [])
        return [r.strip() for r in rules if r and r.strip()]
    except Exception:
        return []


def _rules_suffix() -> str:
    """把经验规则拼成追加到 system prompt 的片段；无规则时为空串。"""
    rules = load_diagnosis_rules()
    if not rules:
        return ""
    lines = ["", "此外，请务必遵守以下从历史诊断错误中总结出的判别原则（优先级高于上面的通用描述）："]
    lines += [f"- {r}" for r in rules]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------
def _diagnose_with_llm(goal: str, concepts: list[Concept], user_text: str) -> DiagnosticResult | None:
    concept_desc = "\n".join(
        f"- {c.id}：{c.name}（{c.summary}）" for c in concepts
    )
    data = chat_json(
        _DIAG_SYSTEM + _rules_suffix(),
        f"学习目标：{goal}\n\n概念列表：\n{concept_desc}\n\n学习者的理解陈述：\n{user_text}",
        temperature=0.2,
        max_tokens=800,
    )
    if not data:
        return None
    try:
        state = data.get("state", "partial")
        if state not in ("understood", "partial", "misconceived", "insufficient"):
            state = "partial"
        return DiagnosticResult(
            state=state,
            confidence=float(data.get("confidence", 0.5)),
            concept_ids=list(data.get("concept_ids", [])),
            evidence=str(data.get("evidence", "")),
            misconception=str(data.get("misconception", "")),
            missing=list(data.get("missing", [])),
        )
    except Exception:
        return None


# 表示"不知道/不会/不清楚"的信号词
_UNKNOWN_HINTS = (
    "不知道", "不清楚", "不了解", "不会", "没学过", "没听过", "不确定", "忘了", "忘记了", "不懂",
    "不太清楚", "不太懂", "没太懂", "太清楚", "只是听说", "听说过", "只知道", "了解不多", "了解一点",
    "一知半解", "说不出", "说不上来", "没概念",
)


def _heuristic_diagnose(concepts: list[Concept], user_text: str, focus_concept_id: str | None) -> DiagnosticResult:
    """无 LLM 时的启发式诊断。诚实标注能力边界，主要依赖否定信号与信息量。

    evidence 字段严格保持「学习者原话要点」语义，只引用原话截断，
    绝不写入引擎自述文案（避免内部机制状态泄漏到回复层）。
    """
    t = user_text.strip()
    low = t.lower()
    snippet = t[:60] if t else ""  # 引用原话要点，而非引擎自述

    # 信息不足：明确否定 / 表达过短
    if any(h in t for h in _UNKNOWN_HINTS) or len(t) < 8:
        return DiagnosticResult(
            state="insufficient",
            confidence=0.85 if len(t) < 8 else 0.9,
            concept_ids=[focus_concept_id] if focus_concept_id else [],
            evidence=snippet,
            misconception="",
            missing=[],
        )

    # 自我纠正信号 → 错误
    if any(h in t for h in ("我以为", "我以为是", "说错了", "搞混了", "记错了", "混淆")):
        return DiagnosticResult(
            state="misconceived",
            confidence=0.7,
            concept_ids=[focus_concept_id] if focus_concept_id else [],
            evidence=snippet,
            misconception="学习者自述存在混淆",
            missing=[],
        )

    # 命中概念关键词，信息量足够 → 半理解（降级模式无法验证语义对错，保守判定）
    name_hits = sum(1 for c in concepts if c.name and c.name in t)
    if name_hits >= 1 and len(t) >= 12:
        return DiagnosticResult(
            state="partial",
            confidence=0.5,
            concept_ids=[focus_concept_id] if focus_concept_id else [],
            evidence=snippet,
            misconception="",
            missing=[],
        )

    # 其余情况默认半理解，交由后续追问继续收集证据
    return DiagnosticResult(
        state="partial",
        confidence=0.4,
        concept_ids=[focus_concept_id] if focus_concept_id else [],
        evidence=snippet,
        misconception="",
        missing=[],
    )


def diagnose(
    goal: str,
    concepts: list[Concept],
    user_text: str,
    focus_concept_id: str | None = None,
) -> DiagnosticResult:
    """诊断用户表达，LLM 优先，降级到启发式。"""
    result = _diagnose_with_llm(goal, concepts, user_text)
    if result is not None:
        return result
    return _heuristic_diagnose(concepts, user_text, focus_concept_id)


# ---------------------------------------------------------------------------
# 用户画像（千人千面）：不同背景的起点与学习速度不同
# ---------------------------------------------------------------------------
# 先验水平 -> BKT 参数
PROFILE_PARAMS = {
    "novice": {
        "label": "新手",
        "P_L0": 0.20,
        "P_LEARN": 0.15,
        "P_GUESS": 0.25,
        "P_SLIP": 0.12,
    },
    "intermediate": {
        "label": "有基础",
        "P_L0": 0.35,
        "P_LEARN": 0.20,
        "P_GUESS": 0.20,
        "P_SLIP": 0.10,
    },
    "advanced": {
        "label": "进阶",
        "P_L0": 0.50,
        "P_LEARN": 0.28,
        "P_GUESS": 0.15,
        "P_SLIP": 0.08,
    },
}

_ADVANCED_HINTS = ("进阶", "深入", "原理", "优化", "性能", "高级", "底层", "源码", "架构", "分布式")
_NOVICE_HINTS = ("入门", "基础", "新手", "初学", "零基础", "小白", "了解", "概览", "是什么")


def infer_prior_level(goal: str) -> str:
    """从学习目标推断先验水平（千人千面的入口）。"""
    low = goal.lower()
    if any(h in low for h in _ADVANCED_HINTS):
        return "advanced"
    if any(h in low for h in _NOVICE_HINTS):
        return "novice"
    return "intermediate"


def get_profile_params(level: str) -> dict:
    """返回某先验水平对应的 BKT 参数（未知水平安全降级到 intermediate）。"""
    return PROFILE_PARAMS.get(level, PROFILE_PARAMS["intermediate"])


# ---------------------------------------------------------------------------
# BKT 贝叶斯更新
# ---------------------------------------------------------------------------
# 认知状态 -> 软正确分数（0 完全错误，1 完全正确）
_STATE_SCORE = {
    "understood": 0.95,
    "partial": 0.6,
    "insufficient": 0.45,   # 中性偏弱，几乎不改变
    "misconceived": 0.12,
}


def bayes_update(
    prior: float,
    state: str,
    confidence: float,
    slip: float | None = None,
    guess: float | None = None,
    learn: float | None = None,
) -> float:
    """根据一次诊断证据，用软化贝叶斯规则更新掌握概率。

    prior: 当前掌握概率 P(mastered)
    state: 认知状态
    confidence: 诊断置信度（用于缩放证据强度）
    slip/guess/learn: 可选，用于千人千面（不传则用全局 config 默认值）
    """
    raw_score = _STATE_SCORE.get(state, 0.5)
    # 低置信度 -> 证据向 0.5 靠拢，削弱更新幅度
    score = 0.5 + (raw_score - 0.5) * max(0.0, min(1.0, confidence))

    slip = config.P_SLIP if slip is None else slip
    guess = config.P_GUESS if guess is None else guess
    learn = config.P_LEARN if learn is None else learn

    # P(observation | mastered) 与 P(observation | not mastered)
    p_obs_mastered = score * (1 - slip) + (1 - score) * slip
    p_obs_not_mastered = score * guess + (1 - score) * (1 - guess)

    num = prior * p_obs_mastered
    den = num + (1 - prior) * p_obs_not_mastered
    if den == 0:
        return prior
    posterior = num / den

    # 学习转移：未掌握者有一定概率通过学习转入掌握
    posterior += (1 - posterior) * learn * max(0.0, min(1.0, confidence)) * 0.3

    return max(0.0, min(1.0, posterior))


def state_from_mastery(mastery: float) -> str:
    """掌握概率 -> 认知状态（用于可视化）。"""
    if mastery >= config.MASTERY_THRESHOLD:
        return "understood"
    if mastery >= config.STATE_BANDS["partial"]:
        return "partial"
    return "insufficient"
