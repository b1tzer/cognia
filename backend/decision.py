"""分层流程控制决策引擎。

落地「确定性骨架 + LLM 语义决策」混合架构（对应架构设计）：

- 骨架层（纯代码，零 token，不可被 AI 突破）：
    拓扑约束、收敛阈值、防死循环安全网、候选集生成。
- 决策层（LLM 可选）：
    在骨架圈定的候选集内做语义选择，输出结构化理由。
- 校验回退：
    LLM 输出不合法时，回退到确定性规则，保证流程永远收敛。

职责拆分（分层单一职责）：
- 教学动作决策 decide_action：针对当前焦点概念，决定 probe/explain/correct/backtrack/advance
- 焦点概念选择 next_focus_concept：决定下一个要学的概念（拓扑约束 + ZPD + 回溯）
"""
from __future__ import annotations

from typing import Any, Optional

import config
from llm import chat_json
from schemas import ActionDecision, ActionReason, DiagnosticResult

# 认知状态四分类的中文标签（决策层自持，避免跨模块依赖）
_STATE_LABEL = {
    "understood": "理解正确",
    "partial": "半理解（有遗漏或模糊）",
    "misconceived": "存在错误理解",
    "insufficient": "信息不足",
}


# ===========================================================================
# 一、教学动作决策（针对当前焦点概念：决定「怎么教」）
# ===========================================================================

def build_action_candidates(
    state: str, evidence_count: int, consecutive_failures: int
) -> list[dict]:
    """根据安全不变量生成合法候选动作集（按优先级降序）。

    这是「确定性外壳」的核心：LLM 只能在此集合内选择，不能自创动作。
    """
    candidates: list[dict] = []
    if state == "misconceived":
        # 连续纠错无效 → 回溯到前置概念重新巩固（防「反复横跳」）
        if consecutive_failures >= config.BACKTRACK_CONSECUTIVE_FAILURES or evidence_count >= 2:
            candidates.append({
                "action": "backtrack",
                "why_eligible": "连续纠错无效，回溯到前置概念重新巩固",
            })
        candidates.append({
            "action": "correct",
            "why_eligible": "存在明确误解，用反例或反问引导其自我发现矛盾",
        })
    elif state == "insufficient":
        if evidence_count >= 2:
            candidates.append({
                "action": "explain",
                "why_eligible": "连续信息不足，直接解释降低挫败感",
            })
        else:
            candidates.append({
                "action": "probe",
                "why_eligible": "信息不足，追问引导其先表达已有理解",
            })
            candidates.append({
                "action": "explain",
                "why_eligible": "若判断其基础薄弱，可改为直接解释",
            })
    elif state == "partial":
        candidates.append({
            "action": "probe",
            "why_eligible": "方向对但有遗漏，追问补全因果链",
        })
        candidates.append({
            "action": "explain",
            "why_eligible": "若遗漏点单一明确，可定向解释",
        })
    elif state == "understood":
        candidates.append({
            "action": "advance",
            "why_eligible": "理解正确，抛迁移性问题检验",
        })
    return candidates


def decide_action_rule(
    state: str, evidence_count: int, consecutive_failures: int
) -> ActionDecision:
    """确定性规则降级：取候选集第一个（按教学优先级排序）。"""
    candidates = build_action_candidates(state, evidence_count, consecutive_failures)
    action = candidates[0]["action"] if candidates else "probe"
    return ActionDecision(
        chosen_action=action,
        reasons=ActionReason(
            criterion_used="确定性规则降级",
            pedagogical_intent="LLM 不可用或输出非法，回退到规则决策",
            confidence=1.0,
        ),
    )


_ACTION_DECIDE_SYSTEM = """你是 Cognia 的教学决策裁判。你只负责从给定候选动作集中选一个，不负责写任何教学文案。

当前焦点概念：{concept}
诊断状态：{state_label}（置信度 {confidence}）
诊断证据：{evidence}
遗漏点：{missing}
误解：{misconception}
认知快照：掌握度 {mastery}，已收集证据 {evidence_count} 条，连续失败 {consecutive_failures} 次

合法候选动作（你只能从中选一个）：
{candidates}

规则：
1. 只能选候选集内列出的动作，禁止自创动作。
2. 理由必须引用上方真实字段（evidence/missing/misconception/mastery），禁止编造。
3. 只输出 JSON，不要任何多余文字。

输出格式：
{{"chosen_action":"probe","reasons":{{"evidence_cited":"...","criterion_used":"...","pedagogical_intent":"...","confidence":0.8}}}}
"""


def decide_action_llm(
    concept_name: str,
    diagnosis: DiagnosticResult,
    candidates: list[dict],
    mastery: float,
    evidence_count: int,
    consecutive_failures: int,
) -> Optional[ActionDecision]:
    """LLM 在候选集内做语义选择，输出结构化理由。失败返回 None。"""
    cand_text = "\n".join(f'- {c["action"]}：{c["why_eligible"]}' for c in candidates)
    system = _ACTION_DECIDE_SYSTEM.format(
        concept=concept_name,
        state_label=_STATE_LABEL.get(diagnosis.state, "半理解"),
        confidence=diagnosis.confidence,
        evidence=diagnosis.evidence or "无",
        missing="、".join(diagnosis.missing) if diagnosis.missing else "无",
        misconception=diagnosis.misconception or "无",
        mastery=f"{mastery:.2f}",
        evidence_count=evidence_count,
        consecutive_failures=consecutive_failures,
        candidates=cand_text,
    )
    data = chat_json(system, "", temperature=0.2, max_tokens=400)
    if not data:
        return None
    try:
        action = data.get("chosen_action", "probe")
        if action not in ("probe", "explain", "correct", "backtrack", "advance"):
            return None
        r = data.get("reasons") or {}
        return ActionDecision(
            chosen_action=action,
            reasons=ActionReason(
                evidence_cited=str(r.get("evidence_cited", "")),
                criterion_used=str(r.get("criterion_used", "")),
                pedagogical_intent=str(r.get("pedagogical_intent", "")),
                confidence=float(r.get("confidence", 0.5)),
            ),
        )
    except Exception:
        return None


def decide_action(
    state: str,
    evidence_count: int,
    consecutive_failures: int = 0,
    concept_name: str = "",
    diagnosis: Optional[DiagnosticResult] = None,
    mastery: float = 0.0,
) -> ActionDecision:
    """教学动作决策主入口：骨架安全网 → LLM 语义决策 → 规则回退。"""
    # 安全网 1（不可被 LLM 覆盖）：单概念步数超限 → 强制回溯，防死循环
    if evidence_count >= config.MAX_STEPS_PER_CONCEPT:
        return ActionDecision(
            chosen_action="backtrack",
            reasons=ActionReason(
                evidence_cited=f"evidence_count={evidence_count}",
                criterion_used="步数上限安全网",
                pedagogical_intent="该概念交互次数过多，强制回溯到前置概念避免死循环",
                confidence=1.0,
            ),
        )

    candidates = build_action_candidates(state, evidence_count, consecutive_failures)

    # LLM 语义决策：候选集内选择
    if config.AI_ENABLED and diagnosis is not None:
        decision = decide_action_llm(
            concept_name, diagnosis, candidates, mastery, evidence_count, consecutive_failures
        )
        # 校验：LLM 输出必须落在候选集内，否则视为非法
        if decision is not None and decision.chosen_action in {c["action"] for c in candidates}:
            return decision

    return decide_action_rule(state, evidence_count, consecutive_failures)


# ===========================================================================
# 二、焦点概念选择（决定「学哪个」）
# ===========================================================================

def topo_order(knowledge: dict) -> list[str]:
    """对概念做拓扑排序，返回按学习顺序排列的概念 id 列表。"""
    concepts = {c["id"]: c for c in knowledge["concepts"]}
    visited: set[str] = set()
    order: list[str] = []

    def visit(cid: str, stack: set[str]):
        if cid in visited:
            return
        if cid in stack:
            return  # 环保护
        stack.add(cid)
        for pre in concepts.get(cid, {}).get("prerequisites", []):
            if pre in concepts:
                visit(pre, stack)
        stack.discard(cid)
        visited.add(cid)
        order.append(cid)

    for c in concepts:
        visit(c, set())
    return order


def zpd_score(mastery: float) -> float:
    """最近发展区（ZPD）权重：越接近掌握阈值但未达，越值得优先推一把。

    返回 0~1，用于候选概念排序（非概率）。半懂区间 [ZPD_MIN, θ) 最高优先。
    """
    theta = config.MASTERY_THRESHOLD
    if mastery >= theta:
        return 0.0  # 已掌握，不应再选（防御）
    zmin = config.ZPD_MIN
    if mastery >= zmin:
        return 1.0  # 半懂（最近发展区核心）
    # 未接触（mastery < zmin）：低优先，随 mastery 升高而升高
    return 0.3 + 0.7 * (mastery / zmin)


def build_focus_candidates(knowledge: dict, cognitive: dict) -> list[dict]:
    """生成合法焦点候选集：拓扑约束（前置已掌握 && 自身未掌握）+ ZPD 评分。"""
    mastery_map = {m["concept_id"]: m for m in cognitive["concepts"]}
    concepts = {c["id"]: c for c in knowledge["concepts"]}
    order = topo_order(knowledge)

    candidates: list[dict] = []
    for cid in order:
        c = concepts.get(cid)
        if not c:
            continue
        mastery = mastery_map.get(cid, {}).get("mastery", 0.0)
        if mastery >= config.MASTERY_THRESHOLD:
            continue
        # 前置概念是否都已基本掌握（拓扑硬约束）
        prereq_ok = all(
            mastery_map.get(p, {}).get("mastery", 0.0) >= config.MASTERY_THRESHOLD
            for p in c.get("prerequisites", [])
        )
        if prereq_ok:
            candidates.append({
                "concept": c,
                "mastery": mastery,
                "zpd_score": zpd_score(mastery),
            })
    return candidates


def weakest_prereq(
    cid: str, concepts: dict, mastery_map: dict
) -> Optional[str]:
    """返回 cid 的前置中掌握度最低的那个；无前置返回 None。

    回溯语义：当前概念反复卡住，通常是某个前置掌握不牢，
    回到最弱前置重新巩固。
    """
    c = concepts.get(cid)
    if not c:
        return None
    prereqs = c.get("prerequisites", [])
    if not prereqs:
        return None
    return min(prereqs, key=lambda p: mastery_map.get(p, {}).get("mastery", 0.0))


def detect_backtrack_target(knowledge: dict, cognitive: dict) -> Optional[str]:
    """检测回溯目标，两种触发：

    1. 某概念连续失败达到阈值 → 回溯到其最弱前置（防「反复横跳」）
    2. 某概念的某前置掌握度衰退到 floor 以下 → 回溯到该前置（防「地基松动」）
    """
    mastery_map = {m["concept_id"]: m for m in cognitive["concepts"]}
    concepts = {c["id"]: c for c in knowledge["concepts"]}

    # 触发 1：连续失败 → 回溯到最弱前置
    for m in cognitive["concepts"]:
        if m.get("consecutive_failures", 0) >= config.BACKTRACK_CONSECUTIVE_FAILURES:
            target = weakest_prereq(m["concept_id"], concepts, mastery_map)
            if target:
                return target

    # 触发 2：前置掌握度衰退 → 回溯到该前置
    for c in knowledge["concepts"]:
        for p in c.get("prerequisites", []):
            if mastery_map.get(p, {}).get("mastery", 0.0) < config.BACKTRACK_MASTERY_FLOOR:
                return p

    return None


_FOCUS_SELECT_SYSTEM = """你是 Cognia 的学习路径规划裁判。给定认知快照和合法候选概念集，选出下一个要聚焦的概念。

学习目标：{goal}

候选概念集（代码已保证前置依赖满足，你只需在其中选择）：
{candidates}

规则：
1. 只能选候选集内的 id，禁止自创。
2. 优先选择最近发展区（ZPD）内的概念：掌握度中等、最接近突破的概念。
3. 只输出 JSON，不要任何多余文字。

输出格式：
{{"selected_concept_id":"...","reason":"...","confidence":0.8}}
"""


def select_focus_llm(
    knowledge: dict, cognitive: dict, candidates: list[dict]
) -> Optional[str]:
    """LLM 在候选集内语义选择焦点概念。失败返回 None。"""
    cand_desc = "\n".join(
        f'- {c["concept"]["id"]}：{c["concept"]["name"]}'
        f'（掌握度 {c["mastery"]:.2f}，ZPD 权重 {c["zpd_score"]:.2f}）'
        for c in candidates
    )
    system = _FOCUS_SELECT_SYSTEM.format(
        goal=knowledge.get("goal", ""), candidates=cand_desc
    )
    data = chat_json(system, "", temperature=0.2, max_tokens=400)
    if not data:
        return None
    cid = data.get("selected_concept_id", "")
    valid = {c["concept"]["id"] for c in candidates}
    return cid if cid in valid else None


def next_focus_concept(knowledge: dict, cognitive: dict) -> Optional[dict]:
    """焦点概念选择主入口：回溯优先 → LLM 语义选择 → 规则回退（ZPD 排序）。"""
    candidates = build_focus_candidates(knowledge, cognitive)
    if not candidates:
        return None

    concepts = {c["id"]: c for c in knowledge["concepts"]}

    # 回溯优先（确定性，不可被 LLM 覆盖）：连续失败 → 回到最上游未掌握前置
    bt = detect_backtrack_target(knowledge, cognitive)
    if bt is not None and bt in concepts:
        return concepts[bt]

    # LLM 语义选择：候选集内选择
    if config.AI_ENABLED:
        chosen = select_focus_llm(knowledge, cognitive, candidates)
        if chosen is not None and chosen in concepts:
            return concepts[chosen]

    # 规则回退：ZPD 权重降序（半懂优先），拓扑序为稳定 tie-breaker
    candidates.sort(key=lambda x: -x["zpd_score"])
    return candidates[0]["concept"]
