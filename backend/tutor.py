"""教学决策引擎 + 苏格拉底式对话生成。

实现 Purpose.md 中的闭环决策：
根据诊断结果（理解/半理解/错误/信息不足），决定下一步动作
（追问 / 解释 / 纠错 / 回溯 / 继续），并用苏格拉底提问引导而非直接给答案。
"""
from __future__ import annotations

from typing import Any, Optional

from llm import chat_text
from schemas import Concept, DiagnosticResult

# ---------------------------------------------------------------------------
# 教学动作决策
# ---------------------------------------------------------------------------
def decide_action(state: str, evidence_count: int) -> str:
    """根据认知状态与证据数量决定教学动作。"""
    if state == "misconceived":
        # 连续误解（纠错无效）→ 回溯到前置概念重新巩固
        if evidence_count >= 2:
            return "backtrack"
        return "correct"
    if state == "insufficient":
        if evidence_count >= 2:
            return "explain"  # 连续信息不足，改为直接解释降低挫败感
        return "probe"
    if state == "partial":
        return "probe"
    if state == "understood":
        return "advance"
    return "probe"


# ---------------------------------------------------------------------------
# LLM 苏格拉底回复
# ---------------------------------------------------------------------------
_TUTOR_SYSTEM = """你是一名苏格拉底式导师（Socratic tutor），目标是帮助学习者真正理解概念。

你现在聚焦于概念「{concept}」。概念说明：{summary}
「真正理解它」意味着：{why_matters}
常见误解：{misconceptions}

学习者刚才的表达被诊断为：{state_label}
你本次的教学动作是：{action_label}

请严格按教学动作执行，**只输出一小段话，不要长篇大论**：
- probe（追问）：针对他模糊/遗漏处提问引导，让他自己补全因果链
- explain（解释）：直接、简明地把这个知识点讲清楚，帮他建立正确理解（此时不要绕弯子提问）
- correct（纠错）：不直接纠错，用一个反例或反问让他自己发现矛盾
- backtrack（回溯）：温和指出当前概念反复卡住，建议先退回到它的前置概念重新巩固，并点明两者关系
- advance（继续）：先肯定，再抛一个更有挑战性的问题检验迁移能力
"""

_STATE_LABEL = {
    "understood": "理解正确",
    "partial": "半理解（有遗漏或模糊）",
    "misconceived": "存在错误理解",
    "insufficient": "信息不足",
}

_ACTION_LABEL = {
    "probe": "追问",
    "explain": "解释",
    "correct": "纠错",
    "backtrack": "回溯",
    "advance": "继续",
}


def _tutor_with_llm(
    concept: Concept,
    diagnosis: DiagnosticResult,
    action: str,
) -> Optional[str]:
    system = _TUTOR_SYSTEM.format(
        concept=concept.name,
        summary=concept.summary or concept.name,
        why_matters=concept.why_matters or "建立准确的心智模型",
        misconceptions="；".join(concept.common_misconceptions) or "暂无记录",
        state_label=_STATE_LABEL.get(diagnosis.state, "半理解"),
        action_label=_ACTION_LABEL.get(action, "追问"),
    )
    user = f"学习者表达：{diagnosis.evidence or ''}"
    if diagnosis.misconception:
        user += f"\n已识别的误解：{diagnosis.misconception}"
    return chat_text(system, user, temperature=0.6, max_tokens=1500)


# ---------------------------------------------------------------------------
# 模板降级回复
# ---------------------------------------------------------------------------
def _tutor_template(concept: Concept, diagnosis: DiagnosticResult, action: str) -> str:
    name = concept.name
    if action == "explain":
        why = f"真正理解它的关键在于：{concept.why_matters}。" if concept.why_matters else ""
        return (
            f"看得出这块对你来说还比较陌生，我先直接讲清楚，帮你把地基打起来。\n"
            f"「{name}」的核心是：{concept.summary or '理解它的机制与因果'}。{why}"
            "听完之后，你用自己的话复述一遍，我来看看你是不是真的懂了。"
        )
    if action == "correct":
        hint = diagnosis.misconception or f"你对「{name}」的理解里可能藏着一个容易忽略的地方"
        return (
            f"有意思，你刚才提到「{name}」时，我留意到“{hint}”。"
            "先不直接告诉你答案——你试着反问自己：如果这个说法成立，会出现什么自相矛盾的结果？"
        )
    if action == "backtrack":
        return (
            f"我们在「{name}」上绕了几次，可能说明更基础的一环还没完全打通。"
            "先不急着往下走——退一步，回到它的前置概念，把那个基础重新理清楚，"
            "再来攻克这里，会顺利很多。"
        )
    if action == "advance":
        return (
            f"很好，你对「{name}」的理解基本到位。既然已经掌握，我们把它放到新场景里检验一下："
            f"如果换一个你之前没见过的情境，{concept.why_matters or '这个机制'}还会成立吗？"
            "试着举一个例子说明它为什么成立（或不成立）。"
        )
    # probe（默认追问）
    if diagnosis.state == "insufficient":
        return (
            f"我们先别急着下结论。关于「{name}」，你能用一两句话说说你已经知道的吗？"
            "哪怕是零散的印象或模糊的理解都可以。"
        )
    return (
        f"你已经抓到了「{name}」的一部分，方向是对的。"
        f"但还有一个关键环节没串起来——{concept.why_matters or '它背后的因果链'}。"
        "你能试着把它和前面学过的东西连起来，说说“为什么会这样”吗？"
    )


def generate_tutor_reply(concept: Concept, diagnosis: DiagnosticResult, action: str) -> str:
    """生成教学回复，LLM 优先，降级到模板。"""
    text = _tutor_with_llm(concept, diagnosis, action)
    if text:
        return text.strip()
    return _tutor_template(concept, diagnosis, action)


# ---------------------------------------------------------------------------
# 开场白 / 学习路径调度
# ---------------------------------------------------------------------------
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


def next_focus_concept(knowledge: dict, cognitive: dict) -> Optional[dict]:
    """选择下一个焦点概念：按拓扑序找第一个未掌握、且前置已掌握的概念。"""
    mastery_map = {m["concept_id"]: m["mastery"] for m in cognitive["concepts"]}
    concepts = {c["id"]: c for c in knowledge["concepts"]}
    order = topo_order(knowledge)

    for cid in order:
        c = concepts.get(cid)
        if not c:
            continue
        if mastery_map.get(cid, 0.0) >= 0.8:
            continue
        # 前置概念是否都已基本掌握
        prereq_ok = all(mastery_map.get(p, 0.0) >= 0.8 for p in c.get("prerequisites", []))
        if prereq_ok:
            return c
    return None


def build_intro(knowledge: dict, focus: Optional[dict]) -> str:
    """开场白：说明已建立的知识模型，并抛出第一个诊断问题。"""
    concepts = knowledge["concepts"]
    if focus is None:
        focus = concepts[0] if concepts else None
    if focus is None:
        return "我已经为你的学习目标建立了知识模型，让我们开始吧。"
    lines = [
        f"我已经把「{knowledge['goal']}」拆解成 {len(concepts)} 个需要理解的概念，"
        "并梳理了它们之间的依赖关系。",
        "",
        f"我们从一个基础概念开始：**{focus['name']}**。",
        f"在正式讲解之前，我想先看看你现在的理解——你能用自己的话说说，"
        f"「{focus['name']}」是什么、解决什么问题吗？不用追求准确，凭感觉说就好。",
    ]
    return "\n".join(lines)
