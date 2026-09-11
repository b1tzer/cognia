"""诊断引擎评测器：度量北极星指标「诊断准确度」，并输出 token 用量观测。

用法（在 backend 目录下运行）：
    python evaluate.py                # 跑 ground-truth 评测 + token 观测
    python evaluate.py --judge        # 额外跑 LLM-as-judge 交叉验证（会消耗 token）
    python evaluate.py --limit 5      # 只跑前 5 条（调试用）

纯标准库实现，不引入额外依赖（契合「节约」约束）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import config
import cognitive as cog
import domain_model
import llm
from schemas import Concept

EVAL_DIR = Path(__file__).resolve().parent / "eval"
SAMPLES_PATH = EVAL_DIR / "samples.json"

STATES = ["understood", "partial", "misconceived", "insufficient"]

STATE_LABEL = {
    "understood": "理解",
    "partial": "半理解",
    "misconceived": "误解",
    "insufficient": "信息不足",
}

# 领域 -> 学习目标（用于诊断 prompt 的「学习目标」字段）
DOMAIN_GOAL = {
    "http": "理解 HTTP 协议",
    "git": "理解 Git 版本控制",
    "python": "理解 Python 核心机制",
}

# ---------------------------------------------------------------------------
# LLM-as-judge 提示词（无 ground truth 场景下的交叉验证）
# ---------------------------------------------------------------------------
_JUDGE_SYSTEM = """你是认知诊断的资深评审专家。给定学习者对某概念的理解陈述，以及一个诊断引擎给出的认知状态判断，请你独立判断该诊断是否正确。

认知状态四分类：
- understood：理解正确、准确、能说清机制/因果，无明显错误
- partial：半理解，方向对但有遗漏、模糊、不完整
- misconceived：存在明确的概念错误或误解
- insufficient：信息不足，表达太空泛或承认不知道

只输出 JSON：
{"correct": true, "reason": "一句话说明判断依据"}
"""

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_samples() -> list[dict]:
    with open(SAMPLES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["samples"]

def build_concepts(domain: str, goal: str) -> list[Concept]:
    """复用 domain_model 的内置模板，保证评测不依赖 LLM 建模型。"""
    concepts = domain_model._DOMAIN_TEMPLATES.get(domain)
    if concepts is None:
        concepts = domain_model._generic_concepts(goal)
    return concepts

# ---------------------------------------------------------------------------
# ground-truth 评测
# ---------------------------------------------------------------------------
def evaluate_sample(sample: dict) -> dict:
    goal = DOMAIN_GOAL.get(sample["domain"], sample["domain"])
    concepts = build_concepts(sample["domain"], goal)
    result = cog.diagnose(goal, concepts, sample["user_text"])
    return {
        "id": sample["id"],
        "domain": sample["domain"],
        "kind": sample.get("kind", "core"),
        "expected": sample["expected_state"],
        "predicted": result.state,
        "confidence": round(result.confidence, 3),
        "correct": result.state == sample["expected_state"],
    }

def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    # 混淆矩阵 [expected][predicted]
    confusion: dict[str, dict[str, int]] = {e: {p: 0 for p in STATES} for e in STATES}
    # 计数
    expected_count: dict[str, int] = {s: 0 for s in STATES}
    predicted_count: dict[str, int] = {s: 0 for s in STATES}
    tp: dict[str, int] = {s: 0 for s in STATES}

    for r in results:
        e, p = r["expected"], r["predicted"]
        confusion[e][p] += 1
        expected_count[e] += 1
        predicted_count[p] += 1
        if e == p:
            tp[e] += 1

    per_state = {}
    for s in STATES:
        per_state[s] = {
            "expected": expected_count[s],
            "predicted": predicted_count[s],
            "tp": tp[s],
            "precision": round(tp[s] / predicted_count[s], 3) if predicted_count[s] else 0.0,
            "recall": round(tp[s] / expected_count[s], 3) if expected_count[s] else 0.0,
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "confusion": confusion,
        "per_state": per_state,
        "errors": [r for r in results if not r["correct"]],
    }

# ---------------------------------------------------------------------------
# LLM-as-judge 交叉验证
# ---------------------------------------------------------------------------
def llm_judge(sample: dict, predicted_state: str, concepts: list[Concept]) -> dict | None:
    concept_desc = "\n".join(f"- {c.name}：{c.summary}" for c in concepts)
    user = (
        f"学习者对概念的陈述：{sample['user_text']}\n\n"
        f"相关概念：\n{concept_desc}\n\n"
        f"诊断引擎的判断：{STATE_LABEL.get(predicted_state, predicted_state)}（{predicted_state}）"
    )
    data = llm.chat_json(_JUDGE_SYSTEM, user, temperature=0.0)
    if not data:
        return None
    return {
        "id": sample["id"],
        "predicted": predicted_state,
        "judge_correct": bool(data.get("correct", False)),
        "reason": str(data.get("reason", "")),
    }

# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------
def print_report(metrics: dict, usage: dict) -> None:
    print("=" * 64)
    print("Cognia 诊断引擎评测报告（ground-truth 基线）")
    print("=" * 64)
    print(f"样本总数：{metrics['total']}   正确：{metrics['correct']}   "
          f"总体准确率：{metrics['accuracy'] * 100:.1f}%")
    print()
    print("每状态表现（precision / recall）：")
    print(f"  {'状态':<10} {'标注':>4} {'预测':>4} {'精确率':>8} {'召回率':>8}")
    for s in STATES:
        ps = metrics["per_state"][s]
        print(f"  {STATE_LABEL[s]:<10} {ps['expected']:>4} {ps['predicted']:>4} "
              f"{ps['precision'] * 100:>7.1f}% {ps['recall'] * 100:>7.1f}%")
    print()
    print("混淆矩阵（行=期望，列=预测）：")
    header = "  " + "".join(f"{STATE_LABEL[s][:2]:>6}" for s in STATES)
    print(header)
    for e in STATES:
        row = "".join(f"{metrics['confusion'][e][p]:>6}" for p in STATES)
        print(f"  {STATE_LABEL[e]:<10}{row}")

    if metrics["errors"]:
        print()
        print(f"预测错误样本（{len(metrics['errors'])} 条）：")
        for r in metrics["errors"]:
            print(f"  - {r['id']}: 期望={STATE_LABEL[r['expected']]} "
                  f"预测={STATE_LABEL[r['predicted']]} (conf={r['confidence']})")

    print()
    print("token 用量观测：")
    if usage["calls"] == 0:
        print("  （本次评测未调用 LLM，可能处于离线诊断模式 AI_ENABLED=0）")
    else:
        print(f"  调用次数：{usage['calls']}")
        print(f"  prompt_tokens：{usage['prompt_tokens']}")
        print(f"  completion_tokens：{usage['completion_tokens']}")
        print(f"  total_tokens：{usage['total_tokens']}")
    print("=" * 64)

# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Cognia 诊断引擎评测")
    parser.add_argument("--judge", action="store_true", help="额外跑 LLM-as-judge 交叉验证")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    args = parser.parse_args()

    samples = load_samples()
    if args.limit > 0:
        samples = samples[: args.limit]

    llm.reset_usage()

    results = [evaluate_sample(s) for s in samples]
    metrics = compute_metrics(results)
    usage = llm.get_usage()

    print_report(metrics, usage)

    if args.judge:
        print()
        print("LLM-as-judge 交叉验证（独立评审诊断是否合理）：")
        judge_results = []
        for s, r in zip(samples, results):
            goal = DOMAIN_GOAL.get(s["domain"], s["domain"])
            concepts = build_concepts(s["domain"], goal)
            j = llm_judge(s, r["predicted"], concepts)
            if j:
                judge_results.append(j)
        if not judge_results:
            print("  （未获得 judge 结果，可能 AI_ENABLED=0）")
        else:
            agree = sum(1 for j in judge_results if j["judge_correct"])
            print(f"  judge 认可率：{agree}/{len(judge_results)}")
            for j in judge_results:
                mark = "✓" if j["judge_correct"] else "✗"
                print(f"  {mark} {j['id']}: 预测={STATE_LABEL[j['predicted']]} | {j['reason']}")

if __name__ == "__main__":
    main()
