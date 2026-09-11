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
_TUTOR_SYSTEM = """你是一名苏格拉底式导师（Socratic tutor），目标是帮助学习者真正理解概念，
而不是直接告诉他答案。你的教学原则：通过提问引导学习者自己发现、修正、深化理解。

你现在聚焦于概念「{concept}」。概念说明：{summary}
「真正理解它」意味着：{why_matters}
常见误解：{misconceptions}

学习者刚才的表达被诊断为：{state_label}

请按状态采取策略，**只输出一小段话（一个核心问题或简短引导），不要长篇大论，不要直接给完整答案**：
- 信息不足：用一个开放式问题引导他说出已知内容（如"你目前对 X 了解多少？"）
- 半理解：针对他模糊/遗漏处追问澄清，帮他补全因果链
- 错误：不直接纠错，用一个反例或反问让他自己发现矛盾
- 理解正确：先肯定，再抛一个更有挑战性的问题检验迁移能力
"""

_STATE_LABEL = {
    "understood": "理解正确",
    "partial": "半理解（有遗漏或模糊）",
    "misconceived": "存在错误理解",
    "insufficient": "信息不足",
}


def _tutor_with_llm(
    concept: Concept,
    diagnosis: DiagnosticResult,
) -> Optional[str]:
    system = _TUTOR_SYSTEM.format(
        concept=concept.name,
        summary=concept.summary or concept.name,
        why_matters=concept.why_matters or "建立准确的心智模型",
        misconceptions="；".join(concept.common_misconceptions) or "暂无记录",
        state_label=_STATE_LABEL.get(diagnosis.state, "半理解"),
    )
    user = f"学习者表达：{diagnosis.evidence or ''}"
    if diagnosis.misconception:
        user += f"\n已识别的误解：{diagnosis.misconception}"
    return chat_text(system, user, temperature=0.6, max_tokens=600)


# ---------------------------------------------------------------------------
# 模板降级回复
# ---------------------------------------------------------------------------
def _tutor_template(concept: Concept, diagnosis: DiagnosticResult) -> str:
    name = concept.name
    state = diagnosis.state
    if state == "insufficient":
        return (
            f"我们先别急着下结论。关于「{name}」，你能用一两句话说说你已经知道的吗？"
            "哪怕是零散的印象或模糊的理解都可以，我会根据你的说法来判断该从哪里入手。"
        )
    if state == "misconceived":
        hint = diagnosis.misconception or f"你对「{name}」的理解里可能藏着一个容易忽略的地方"
        return (
            f"有意思，你刚才提到「{name}」时，我留意到“{hint}”。"
            "先不直接告诉你答案——你试着反问自己：如果这个说法成立，会出现什么自相矛盾的结果？"
            "或者，你能不能举一个反例来检验它？"
        )
    if state == "partial":
        return (
            f"你已经抓到了「{name}」的一部分，方向是对的。"
            f"但还有一个关键环节没串起来——{concept.why_matters or '它背后的因果链'}。"
            "你能试着把它和前面学过的东西连起来，说说“为什么会这样”吗？"
        )
    # understood
    return (
        f"很好，你对「{name}」的理解基本到位。既然已经掌握，我们把它放到新场景里检验一下："
        f"如果换一个你之前没见过的情境，{concept.why_matters or '这个机制'}还会成立吗？"
        "试着举一个例子说明它为什么成立（或不成立）。"
    )


def generate_tutor_reply(concept: Concept, diagnosis: DiagnosticResult) -> str:
    """生成苏格拉底式回复，LLM 优先，降级到模板。"""
    text = _tutor_with_llm(concept, diagnosis)
    if text:
        return text.strip()
    return _tutor_template(concept, diagnosis)


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
