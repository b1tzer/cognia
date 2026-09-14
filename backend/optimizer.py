"""在线 Prompt 优化反馈回路引擎（慢循环）。

北极星指标「诊断准确度随对话自我迭代」的在线执行器，**数据增量驱动**：
- optimize_loop 作为「检查节拍」定期唤醒，但每次先判断自上次优化以来
  「新增对话轮次」是否达到阈值（OPTIMIZER_MIN_NEW_TURNS），未达到则什么都不做
- 达到阈值后，仅消费**新增的那批消息**（水位线 watermark 记录各会话已消费位置），
  从 SQLite 真实对话中提取各层样本（替代手动 eval/samples.json）
- 用 LLM-as-judge 做无标注质量评估
- 对判错的样本提炼优化规则，注入对应层 prompt
- 质量门：注入后复评认可率上升才保留，否则回滚（保证不退化）

分层单一职责：本模块只做「采集 → 评估 → 提炼 → 质量门 → 注入」编排，
具体层内 prompt 拼接由各层模块通过 prompt_rules.rules_suffix() 完成。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
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
# 需求 #71 重构后，教学动作已与诊断合并（diagnose_and_decide 一次产出 state+action），
# 不再有独立的 decision_action 层可优化。
OPTIMIZABLE_LAYERS = ("diagnosis", "tutor", "domain_model")

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


# ---------------------------------------------------------------------------
# 水位线（watermark）状态：记录各会话已消费到的消息位置
# ---------------------------------------------------------------------------
def _state_path() -> Path:
    return Path(config.OPTIMIZER_STATE_PATH)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict:
    """加载水位状态。文件缺失/损坏时安全返回初始状态。"""
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "last_optimized_at": "", "watermark": {}}
    return {
        "version": data.get("version", 1),
        "last_optimized_at": data.get("last_optimized_at", ""),
        "watermark": data.get("watermark", {}) or {},
    }


def save_state(state: dict) -> None:
    """保存水位状态（原子写：先写临时文件再替换）。"""
    path = _state_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _recent_sessions_safe(limit: int) -> list[dict]:
    """安全读取最近会话；数据库异常时返回空列表（后台任务不崩溃）。"""
    try:
        return db.recent_sessions(limit)
    except Exception:
        logger.exception("optimizer: db.recent_sessions 失败")
        return []


def count_new_user_turns(sessions: list[dict], watermark: dict) -> int:
    """统计自上次水位以来新增的用户消息（对话轮次）数。"""
    total = 0
    for s in sessions:
        start = max(0, watermark.get(s["id"], 0))
        msgs = s.get("messages", [])
        total += sum(1 for m in msgs[start:] if m.get("role") == "user")
    return total


def advance_watermark(sessions: list[dict], watermark: dict) -> dict:
    """把水位推进到每个会话当前的末尾（标记这批消息已被消费）。"""
    new_wm = dict(watermark)
    for s in sessions:
        new_wm[s["id"]] = len(s.get("messages", []))
    return new_wm


def extract_samples(layer: str, limit: int, watermark: dict | None = None) -> list[dict]:
    """从真实对话中提取某层样本（最多 limit 条），只提取水位之后的新消息。

    诊断层：用户消息（携带诊断结果）
    决策层：助手消息（携带 action 与诊断结果）
    回复层：助手消息（携带 reply 与诊断结果）
    知识模型层：每个「新会话」的知识模型（概念图不随对话变化，只在会话首次出现时提取一次）
    """
    watermark = watermark or {}
    samples: list[dict] = []
    sessions = _recent_sessions_safe(limit * 2)  # 多取会话，消息可产多样本
    for s in sessions:
        goal = s.get("goal", "")
        knowledge = s.get("knowledge") or {}
        cognitive = s.get("cognitive") or {}
        start = max(0, watermark.get(s["id"], 0))
        msgs = s.get("messages", [])

        if layer == "domain_model":
            # 新会话（尚未被消费过）才提取其知识模型
            if s["id"] not in watermark and knowledge.get("concepts"):
                samples.append({
                    "layer": layer,
                    "session_id": s["id"],
                    "goal": goal,
                    "concepts": knowledge.get("concepts", []),
                    "root_concepts": knowledge.get("root_concepts", []),
                })
            continue

        for msg in msgs[start:]:
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
def run_cycle(layer: str, limit: int | None = None, watermark: dict | None = None) -> dict:
    """对单层执行一轮完整优化：采集 → 评估 → 提炼 → 质量门 → 注入/回滚。

    仅消费水位之后的新消息。返回本轮结果摘要（供日志与测试断言）。
    """
    limit = limit or config.OPTIMIZER_MAX_SAMPLES_PER_LAYER
    samples = extract_samples(layer, limit, watermark)
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
    """对所有可优化层执行一轮优化，返回汇总。

    数据增量门控：先统计自上次优化以来的新增对话轮次，未达阈值则本轮
    什么都不做（不调用任何 LLM）；达到阈值才消费新增数据，并在结束后
    推进水位，避免下轮重复消费同一批消息。
    """
    limit = limit or config.OPTIMIZER_MAX_SAMPLES_PER_LAYER
    state = load_state()
    watermark = state.get("watermark", {})

    sessions = _recent_sessions_safe(limit * 2)
    new_turns = count_new_user_turns(sessions, watermark)

    # 数据增量门控：新增对话轮次不足 → 不执行任何优化
    if new_turns < config.OPTIMIZER_MIN_NEW_TURNS:
        return {
            "status": "waiting_for_data",
            "new_turns": new_turns,
            "threshold": config.OPTIMIZER_MIN_NEW_TURNS,
            "layers": {},
        }

    summary: dict[str, Any] = {
        "status": "executed",
        "new_turns": new_turns,
        "layers": {},
        "optimized": 0,
        "rolled_back": 0,
    }
    for layer in OPTIMIZABLE_LAYERS:
        result = run_cycle(layer, limit, watermark)
        summary["layers"][layer] = result
        if result.get("status") == "optimized":
            summary["optimized"] += 1
        elif result.get("status") == "rolled_back":
            summary["rolled_back"] += 1

    # 推进水位：无论本轮各层优化是否采纳，这批新消息都已被评估消费
    state["watermark"] = advance_watermark(sessions, watermark)
    state["last_optimized_at"] = _now_iso()
    save_state(state)
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

    logger.info(
        "optimizer loop started (check_interval=%ss, min_new_turns=%s)",
        config.OPTIMIZER_INTERVAL_SECONDS, config.OPTIMIZER_MIN_NEW_TURNS,
    )
    while True:
        started = time.time()
        try:
            summary = await asyncio.to_thread(run_all_cycles)
            if summary.get("status") == "executed":
                logger.info("optimizer cycle executed: %s", summary)
            else:
                # 数据增量未达标，本轮空转（不调用 LLM）
                logger.debug("optimizer waiting for data: %s", summary)
        except Exception:
            logger.exception("optimizer cycle failed")
        elapsed = time.time() - started
        await asyncio.sleep(max(0.0, config.OPTIMIZER_INTERVAL_SECONDS - elapsed))
