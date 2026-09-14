"""焦点概念选择引擎（拓扑约束 + ZPD + 回溯）。

需求 #71 重构后，本模块不再负责「教学动作决策」——该职责已交还 LLM，
由 cognitive.diagnose_and_decide 在自由上下文中输出教学动作（probe/explain/
correct/backtrack/advance）。本模块只保留「物理护栏与事实提供」：

- 焦点概念选择 next_focus_concept：决定下一个要学的概念（拓扑约束 + ZPD + 回溯）
- 用户元指令识别：detect_advance_intent / detect_help_intent（零 token 规则，高于 LLM 语义）
- 拓扑与掌握判定工具：topo_order / is_mastered / zpd_score / detect_backtrack_target
"""
from __future__ import annotations

from typing import Optional

import config
from llm import chat_json
import prompt_rules





# ===========================================================================
# 推进意图识别（确定性规则，零 token）
# ===========================================================================
# 用户明确表达「继续/下一个」等推进信号时，属于「元对话指令」而非「对概念的
# 理解陈述」。若交给诊断层，会被误判 insufficient 并陷入反复解释同一概念的
# 死循环。这里用纯规则识别推进意图，由流程控制层直接路由到 advance。
_ADVANCE_HINTS = (
    "继续下一个", "下一个概念", "下一个", "继续吧", "接着讲", "接着", "往下",
    "讲下一个", "进入下一个",
)


def detect_advance_intent(user_text: str) -> bool:
    """检测用户是否明确表达「推进到下一个概念」的意图。

    仅短句 + 命中推进信号词才触发；长文本（真正的理解陈述）不触发，避免误伤。
    """
    t = (user_text or "").strip()
    if not t or len(t) > 20:
        return False
    return any(h in t for h in _ADVANCE_HINTS)

# 求助意图的「强信号」：明确请求讲解/举例，或自述零基础。命中即触发，不限制长度。
# 这些是「教学意图」而非「诊断证据」，绝不能让诊断层把「我不懂」误判为 insufficient
# 并继续追问（那正是零基础用户卡死的根因）。
_HELP_STRONG_HINTS = (
    "解释一下", "给我讲讲", "讲讲", "讲一下", "详细讲", "再说说",
    "举个例子", "举个", "举例", "没学过", "没听过", "没接触过",
    "零基础", "从零", "完全不懂", "完全不知道", "一点都不懂",
)

# 求助意图的「弱信号」：口语化「不懂/不知道」。仅短句命中才触发，避免误伤正常陈述。
# 注意不收录「不会」——它会误伤「其他线程不会立马知道」这类正常否定表述。
_HELP_WEAK_HINTS = (
    "不懂", "不太懂", "没懂", "没太懂", "不知道", "不了解",
)

def detect_help_intent(user_text: str) -> bool:
    """检测用户是否表达「求助/请求讲解」意图。

    与 detect_advance_intent 对称：求助是「元对话指令」而非「对概念的理解陈述」，
    不应交给诊断层判 insufficient（否则用户说「我不懂」反而被追问）。
    强信号（明确要求讲解/自述零基础）命中即触发；弱信号（口语化不懂）仅短句触发。
    """
    t = (user_text or "").strip()
    if not t:
        return False
    if any(h in t for h in _HELP_STRONG_HINTS):
        return True
    if len(t) > 20:
        return False
    return any(h in t for h in _HELP_WEAK_HINTS)


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


def is_mastered(m: dict) -> bool:
    """概念是否已完成学习（推进的权威标记）。

    mastered 布尔由主流程在「LLM 判定 understood / 用户明确推进 / 步数上限」
    时置 True。旧的三层数值 AND（mastery + success_count + quality）因跨层
    契约易断（生产方只写 mastery、消费方却要求三者齐备）而废弃，改为单一布尔
    标记。数值字段（mastery/success_count/quality）仍维护，仅供前端星图与
    跨会话先验，不再参与推进判定。
    """
    return bool(m.get("mastered", False))


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
        if is_mastered(mastery_map.get(cid, {})):
            continue
        # 前置概念是否都已基本掌握（拓扑硬约束，三层判定）
        prereq_ok = all(
            is_mastered(mastery_map.get(p, {}))
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
    # 仅「学过但衰退」（有证据记录）才回溯；从未学过（evidence_count==0）是正常初始态，
    # 其 mastery 仍是初始值（低于 floor），不应被误判为「衰退」而强行跳焦点。
    # 已 mastered 的前置不回溯：mastered 是「完成/跳过」的权威标记，即使其 mastery
    # 数值低（如用户明确推进意图跳过），也不应再被拉回（否则推进后又退回，形成死循环）。
    for c in knowledge["concepts"]:
        for p in c.get("prerequisites", []):
            pm = mastery_map.get(p, {})
            if pm.get("evidence_count", 0) > 0 and pm.get("mastery", 0.0) < config.BACKTRACK_MASTERY_FLOOR:
                if is_mastered(pm):
                    continue
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
    knowledge: dict, cognitive: dict, candidates: list[dict], trace: list | None = None
) -> Optional[str]:
    """LLM 在候选集内语义选择焦点概念。失败返回 None。"""
    cand_desc = "\n".join(
        f'- {c["concept"]["id"]}：{c["concept"]["name"]}'
        f'（掌握度 {c["mastery"]:.2f}，ZPD 权重 {c["zpd_score"]:.2f}）'
        for c in candidates
    )
    system = _FOCUS_SELECT_SYSTEM.format(
        goal=knowledge.get("goal", ""), candidates=cand_desc
    ) + prompt_rules.rules_suffix("decision_action")
    data = chat_json(system, "", temperature=0.2, max_tokens=1000, trace=trace, trace_label="焦点选择", budget_label="focus_select")
    if not data:
        return None
    cid = data.get("selected_concept_id", "")
    valid = {c["concept"]["id"] for c in candidates}
    return cid if cid in valid else None


def next_focus_concept(knowledge: dict, cognitive: dict, trace: list | None = None) -> Optional[dict]:
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
        chosen = select_focus_llm(knowledge, cognitive, candidates, trace)
        if chosen is not None and chosen in concepts:
            return concepts[chosen]

    # 规则回退：ZPD 权重降序（半懂优先），拓扑序为稳定 tie-breaker
    candidates.sort(key=lambda x: -x["zpd_score"])
    return candidates[0]["concept"]
