"""在线 Prompt 优化反馈回路引擎（慢循环）。

北极星指标「诊断准确度随对话自我迭代」的在线执行器：
- 后台周期任务（optimize_loop）定期触发
- 从 SQLite 真实对话中提取各层样本（替代手动 eval/samples.json）
- 用 LLM-as-judge 做无标注质量评估
- 对判错的样本提炼优化规则，注入对应层 prompt
- 质量门：注入后复评认可率上升才保留，否则回滚（保证不退化）

分层单一职责：本模块只做「采集 → 评估 → 提炼 → 质量门 → 注入」编排，
具体层内 prompt 拼接由各层模块通过 prompt_rules.rules_suffix() 完成。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import config
import db
import llm
import prompt_rules
from schemas import Concept, DiagnosticResult

logger = logging.getLogger("cognia.optimizer")

# 认知状态四分类中文标签（诊断/决策/回复层共用，避免跨模块依赖）
STATE_LABEL = {
    "understood": "理解正确",
    "partial": "半理解（有遗漏或模糊）",
    "misconceived": "存在错误理解",
    "insufficient": "信息不足",
}

ACTION_LABEL = {
    "probe": "追问",
    "explain": "解释",
    "correct": "纠错",
    "backtrack": "回溯",
    "advance": "继续",
}

# 各层可优化顺序（执行顺序）
OPTIMIZABLE_LAYERS = ("diagnosis", "decision_action", "tutor", "domain_model")

# ---------------------------------------------------------------------------
# 无标注质量评估（LLM-as-judge）：每层一个独立评审提示词
# ---------------------------------------------------------------------------
_JUDGE_SYSTEMS = {
    "diagnosis": """你是认知诊断的资深评审专家。给定学习者对某概念的理解陈述，以及诊断引擎给出的认知状态判断，请独立判断该诊断是否正确。

认知状态四分类：
- understood：理解正确、准确、能说清机制/因果，无明显错误
- partial：半理解，方向对但有遗漏、模糊、不完整
- misconceived：存在明确的概念错误或误解
- insufficient：信息不足，表达太空泛或承认不知道

只输出 JSON：{"correct": true, "reason": "一句话说明判断依据"}""",

    "decision_action": """你是教学决策的资深评审专家。给定当前认知状态与教学动作，请独立判断该动作是否合理、符合教学规律。

教学动作：probe(追问) / explain(解释) / correct(纠错) / backtrack(回溯) / advance(继续)。
判断标准：misconceived 应纠错或回溯；insufficient 应追问或解释；partial 应追问或解释；understood 应继续推进。

只输出 JSON：{"correct": true, "reason": "一句话说明判断依据"}""",

    "tutor": """你是教学回复质量的资深评审专家。给定概念、认知状态、教学动作与导师生成的回复，请独立判断该回复是否恰当执行了该教学动作（简洁、聚焦、符合动作语义，而非偏离动作）。

只输出 JSON：{"correct": true, "reason": "一句话说明判断依据"}""",

    "domain_model": """你是课程设计的资深评审专家。给定学习目标与生成的概念依赖图，请独立判断该概念图是否合理（概念覆盖目标、依赖关系正确、无环、粒度适中）。

只输出 JSON：{"correct": true, "reason": "一句话说明判断依据"}""",
}

# ---------------------------------------------------------------------------
# 提炼优化规则（每层一个分析提示词）
# ---------------------------------------------------------------------------
_ANALYZE_SYSTEMS = {
    "diagnosis": """你是认知诊断的资深专家。一个诊断引擎在以下多条真实对话样本上判断错误。请综合分析共性根因，提炼 1~3 条通用、可操作的判别原则，用来修正诊断、避免再犯同类错误。

要求：通用（不针对个案）、可操作、简洁（每条一句话）、数量克制（宁缺毋滥）。
只输出 JSON：{"rules": ["原则1", "原则2"]}""",

    "decision_action": """你是教学决策的资深专家。一个教学决策引擎在以下多条真实对话样本上选择了不合理的动作。请综合分析共性根因，提炼 1~3 条通用、可操作的动作选择原则。

要求：通用、可操作、简洁、数量克制。
只输出 JSON：{"rules": ["原则1", "原则2"]}""",

    "tutor": """你是教学文案的资深专家。一个教学回复引擎在以下多条真实对话样本上回复不当。请综合分析共性根因，提炼 1~3 条通用、可操作的回复原则。

要求：通用、可操作、简洁、数量克制。
只输出 JSON：{"rules": ["原则1", "原则2"]}""",

    "domain_model": """你是课程设计的资深专家。一个知识模型构建器在以下多条真实样本上生成了不合理的概念图。请综合分析共性根因，提炼 1~3 条通用、可操作的概念图设计原则。

要求：通用、可操作、简洁、数量克制。
只输出 JSON：{"rules": ["原则1", "原则2"]}""",
}


# ---------------------------------------------------------------------------
# 样本提取：从真实对话（SQLite）中抽取各层样本
# ---------------------------------------------------------------------------
def _find_concept(knowledge: dict, concept_ids: list[str]) -> dict | None:
    """从知识模型中按 id 找概念；找不到返回 None。"""
    if not concept_ids or not knowledge:
        return None
    by_id = {c["id"]: c for c in knowledge.get("concepts", [])}
    for cid in concept_ids:
        if cid in by_id:
            return by_id[cid]
    return None


def _mastery_of(cognitive: dict, concept_id: str | None) -> dict:
    """从认知模型中取某概念的掌握信息；找不到返回空 dict。"""
    if not cognitive or not concept_id:
        return {}
    for m in cognitive.get("concepts", []):
        if m.get("concept_id") == concept_id:
            return m
    return {}


def extract_samples(layer: str, limit: int) -> list[dict]:
    """从真实对话中提取某层样本（最多 limit 条）。

    诊断层：用户消息（携带诊断结果）
    决策层：助手消息（携带 action 与诊断结果）
    回复层：助手消息（携带 reply 与诊断结果）
    知识模型层：每个会话的知识模型
    """
    samples: list[dict] = []
    try:
        sessions = db.recent_sessions(limit * 2)  # 多取会话，消息可产多样本
    except Exception:
        logger.exception("extract_samples: db.recent_sessions 失败，本轮跳过")
        return samples
    for s in sessions:
        goal = s.get("goal", "")
        knowledge = s.get("knowledge") or {}
        cognitive = s.get("cognitive") or {}

        if layer == "domain_model":
            if knowledge.get("concepts"):
                samples.append({
                    "layer": layer,
                    "session_id": s["id"],
                    "goal": goal,
                    "concepts": knowledge.get("concepts", []),
                    "root_concepts": knowledge.get("root_concepts", []),
                })
            continue

        for msg in s.get("messages", []):
            if len(samples) >= limit:
                return samples
            diag = msg.get("diagnosis") or {}

            if layer == "diagnosis":
                if msg.get("role") == "user" and msg.get("content") and diag.get("state"):
                    samples.append({
                        "layer": layer,
                        "session_id": s["id"],
                        "goal": goal,
                        "concepts": knowledge.get("concepts", []),
                        "user_text": msg["content"],
                        "predicted_state": diag["state"],
                    })
            elif layer == "decision_action":
                if msg.get("role") == "assistant" and msg.get("action") and diag.get("state"):
                    concept = _find_concept(knowledge, diag.get("concept_ids", []))
                    cid = (diag.get("concept_ids") or [None])[0]
                    m = _mastery_of(cognitive, cid)
                    samples.append({
                        "layer": layer,
                        "session_id": s["id"],
                        "goal": goal,
                        "concept_name": (concept or {}).get("name", ""),
                        "state": diag["state"],
                        "action": msg["action"],
                        "evidence_count": m.get("evidence_count", 0),
                        "consecutive_failures": m.get("consecutive_failures", 0),
                        "mastery": m.get("mastery", 0.0),
                        "diagnosis": diag,
                    })
            elif layer == "tutor":
                if msg.get("role") == "assistant" and msg.get("content") and msg.get("action") and diag.get("state"):
                    concept = _find_concept(knowledge, diag.get("concept_ids", []))
                    samples.append({
                        "layer": layer,
                        "session_id": s["id"],
                        "goal": goal,
                        "concept": concept or {"id": "", "name": "", "summary": ""},
                        "state": diag.get("state", "partial"),
                        "action": msg["action"],
                        "reply": msg["content"],
                        "diagnosis": diag,
                    })
    return samples


# ---------------------------------------------------------------------------
# 格式化：judge / analyze 的用户输入
# ---------------------------------------------------------------------------
def _concepts_text(concepts: list[dict], limit: int = 12) -> str:
    lines = []
    for c in concepts[:limit]:
        name = c.get("name", c.get("id", ""))
        summary = c.get("summary", "")
        lines.append(f"- {c.get('id', '')}：{name}（{summary}）")
    return "\n".join(lines) if lines else "（无概念信息）"


def _judge_user(layer: str, sample: dict) -> str:
    goal = sample.get("goal", "")
    if layer == "diagnosis":
        state = sample.get("predicted_state", "")
        return (
            f"学习目标：{goal}\n\n"
            f"概念列表：\n{_concepts_text(sample.get('concepts', []))}\n\n"
            f"学习者的理解陈述：{sample.get('user_text', '')}\n\n"
            f"诊断引擎的判断：{STATE_LABEL.get(state, state)}（{state}）"
        )
    if layer == "decision_action":
        state = sample.get("state", "")
        action = sample.get("action", "")
        return (
            f"学习目标：{goal}\n"
            f"当前焦点概念：{sample.get('concept_name', '') or '（未知）'}\n"
            f"认知状态：{STATE_LABEL.get(state, state)}（{state}）\n"
            f"已收集证据：{sample.get('evidence_count', 0)} 条\n"
            f"引擎选择的动作：{ACTION_LABEL.get(action, action)}（{action}）"
        )
    if layer == "tutor":
        state = sample.get("state", "")
        action = sample.get("action", "")
        concept = sample.get("concept", {})
        return (
            f"学习目标：{goal}\n"
            f"概念：{concept.get('name', '')}（{concept.get('summary', '')}）\n"
            f"认知状态：{STATE_LABEL.get(state, state)}\n"
            f"教学动作：{ACTION_LABEL.get(action, action)}（{action}）\n"
            f"导师回复：{sample.get('reply', '')}"
        )
    # domain_model
    return (
        f"学习目标：{goal}\n\n"
        f"生成的概念依赖图：\n{_concepts_text(sample.get('concepts', []))}\n"
        f"根概念：{', '.join(sample.get('root_concepts', [])) or '（无）'}"
    )


def _analyze_user(layer: str, errors: list[dict]) -> str:
    parts = []
    for i, s in enumerate(errors, 1):
        parts.append(f"样本{i}：\n{_judge_user(layer, s)}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 单样本评估（无标注 LLM-as-judge）
# ---------------------------------------------------------------------------
def judge_sample(layer: str, sample: dict) -> bool:
    """独立评审该层样本的输出是否合理。LLM 不可用时保守返回 True（不误判为错误）。"""
    data = llm.chat_json(_JUDGE_SYSTEMS[layer], _judge_user(layer, sample), temperature=0.0, max_tokens=200)
    if not data:
        return True
    return bool(data.get("correct", False))


def generate_rules(layer: str, errors: list[dict]) -> list[str]:
    """聚合分析错误样本，提炼优化规则。失败返回空列表。"""
    if not errors:
        return []
    data = llm.chat_json(_ANALYZE_SYSTEMS[layer], _analyze_user(layer, errors), temperature=0.3, max_tokens=800)
    if not data:
        return []
    rules = data.get("rules", [])
    if isinstance(rules, str):
        rules = [rules]
    return [r.strip() for r in rules if r and r.strip()]


# ---------------------------------------------------------------------------
# 复评（质量门用）：注入新规则后重新执行该层，返回新的预测字段
# ---------------------------------------------------------------------------
def reevaluate(layer: str, sample: dict) -> dict:
    """注入新规则后重新执行该层，返回新的预测字段 {字段名: 新值}。

    失败或无法复评返回空 dict（保守：视为无法证明改善，触发回滚）。
    """
    try:
        if layer == "diagnosis":
            import cognitive
            concepts = [Concept(**c) for c in sample.get("concepts", [])]
            result = cognitive.diagnose(sample["goal"], concepts, sample["user_text"])
            return {"predicted_state": result.state}
        if layer == "decision_action":
            import decision
            diag = DiagnosticResult(**sample.get("diagnosis", {}))
            d = decision.decide_action(
                sample["state"],
                sample.get("evidence_count", 0),
                sample.get("consecutive_failures", 0),
                concept_name=sample.get("concept_name", ""),
                diagnosis=diag,
                mastery=sample.get("mastery", 0.0),
            )
            return {"action": d.chosen_action}
        if layer == "tutor":
            import tutor
            concept = Concept(**sample.get("concept", {}))
            diag = DiagnosticResult(**sample.get("diagnosis", {}))
            reply = tutor.generate_tutor_reply(concept, diag, sample["action"])
            return {"reply": reply}
        if layer == "domain_model":
            import domain_model
            km = domain_model.build_knowledge_model(sample["goal"])
            dump = km.model_dump()
            return {"concepts": dump["concepts"], "root_concepts": dump["root_concepts"]}
    except Exception:
        logger.exception("reevaluate failed for layer=%s", layer)
    return {}


# ---------------------------------------------------------------------------
# 单层一轮优化
# ---------------------------------------------------------------------------
def run_cycle(layer: str, limit: int | None = None) -> dict:
    """对单层执行一轮完整优化：采集 → 评估 → 提炼 → 质量门 → 注入/回滚。

    返回本轮结果摘要（供日志与测试断言）。
    """
    limit = limit or config.OPTIMIZER_MAX_SAMPLES_PER_LAYER
    samples = extract_samples(layer, limit)
    if len(samples) < config.OPTIMIZER_MIN_SAMPLES:
        return {
            "layer": layer,
            "status": "skipped",
            "reason": f"样本不足（{len(samples)} < {config.OPTIMIZER_MIN_SAMPLES}）",
            "samples": len(samples),
        }

    # 1. baseline 评估
    judged = [judge_sample(layer, s) for s in samples]
    errors = [s for s, ok in zip(samples, judged) if not ok]
    if not errors:
        return {"layer": layer, "status": "no_error", "reason": "无错误样本", "samples": len(samples)}

    # 2. 提炼候选规则（去重）
    candidates = generate_rules(layer, errors)
    old_rules = prompt_rules.load_rules(layer)
    new_rules = [r for r in candidates if r not in old_rules]
    if not new_rules:
        return {"layer": layer, "status": "no_new_rule", "reason": "未提炼出新规则", "samples": len(samples)}

    # 3. 质量门：注入新规则 → 复评错误样本 → 认可率上升才保留，否则回滚
    prompt_rules.save_rules(layer, old_rules + new_rules)
    baseline_correct = sum(judged)
    after_correct = baseline_correct
    for s in errors:
        new_fields = reevaluate(layer, s)
        if not new_fields:
            continue
        s2 = dict(s)
        s2.update(new_fields)
        if judge_sample(layer, s2):
            after_correct += 1

    if after_correct > baseline_correct:
        # 保留新规则
        return {
            "layer": layer,
            "status": "optimized",
            "added": new_rules,
            "correct_before": baseline_correct,
            "correct_after": after_correct,
            "samples": len(samples),
        }
    # 回滚
    prompt_rules.save_rules(layer, old_rules)
    return {
        "layer": layer,
        "status": "rolled_back",
        "reason": f"质量门未通过（{after_correct} <= {baseline_correct}）",
        "samples": len(samples),
    }


def run_all_cycles(limit: int | None = None) -> dict:
    """对所有可优化层执行一轮优化，返回汇总。"""
    summary: dict[str, Any] = {"layers": {}, "optimized": 0, "rolled_back": 0}
    for layer in OPTIMIZABLE_LAYERS:
        result = run_cycle(layer, limit)
        summary["layers"][layer] = result
        if result.get("status") == "optimized":
            summary["optimized"] += 1
        elif result.get("status") == "rolled_back":
            summary["rolled_back"] += 1
    return summary


# ---------------------------------------------------------------------------
# 后台周期调度
# ---------------------------------------------------------------------------
async def optimize_loop() -> None:
    """后台慢循环：定期执行一轮在线优化。

    仅在 AI 引擎启用且优化开关打开时运行；异常不中断循环（记录后继续）。
    """
    if not (config.AI_ENABLED and config.OPTIMIZER_ENABLED):
        logger.info("optimizer disabled (AI_ENABLED=%s, OPTIMIZER_ENABLED=%s)",
                    config.AI_ENABLED, config.OPTIMIZER_ENABLED)
        return

    logger.info("optimizer loop started (interval=%ss)", config.OPTIMIZER_INTERVAL_SECONDS)
    while True:
        started = time.time()
        try:
            summary = await asyncio.to_thread(run_all_cycles)
            logger.info("optimizer cycle done: %s", summary)
        except Exception:
            logger.exception("optimizer cycle failed")
        elapsed = time.time() - started
        await asyncio.sleep(max(0.0, config.OPTIMIZER_INTERVAL_SECONDS - elapsed))
