"""诊断自我迭代器：跑评测 → 分析错误 → 生成判别规则 → 注入 → 复评对比。

这是北极星指标「诊断准确度随对话自我迭代」的执行引擎。
用法（在 backend 目录下运行）：
    python iterate.py                 # 默认迭代 2 轮
    python iterate.py --rounds 3      # 迭代 3 轮
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cognitive
import evaluate
import llm

RULES_PATH = Path(__file__).resolve().parent / "eval" / "diagnosis_rules.json"

# 分析错误样本、提炼共性判别原则的提示词（聚合式：一次性看多条错误）
_ANALYZE_SYSTEM = """你是认知诊断的资深专家。一个诊断引擎在以下多条样本上都判断错了。请你综合分析它们的共性根因，提炼出 1~3 条通用、可操作的判别原则，用来修正诊断引擎、避免再犯同类错误。

判别原则要求：
- 通用：不针对某个具体案例，而是覆盖一整类情况
- 可操作：诊断时能直接据此做判断
- 简洁：每条一句话说清
- 数量克制：只提炼真正必要的原则，宁缺毋滥

只输出 JSON：{"rules": ["原则1", "原则2", ...]}
"""


def load_rules() -> list[str]:
    try:
        data = json.loads(RULES_PATH.read_text(encoding="utf-8"))
        rules = data.get("rules", [])
        return [r.strip() for r in rules if r and r.strip()]
    except Exception:
        return []


def save_rules(rules: list[str]) -> None:
    payload = {
        "version": 1,
        "description": "诊断经验库：从历史诊断错误中总结出的判别原则，注入诊断 prompt 以持续修正（北极星「自我迭代」的载体）。由 iterate.py 自动生成与更新。",
        "rules": rules,
    }
    RULES_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_eval(samples: list[dict]) -> list[dict]:
    """跑完整评测，保留 user_text 以便后续分析错误根因。"""
    results = []
    for s in samples:
        goal = evaluate.DOMAIN_GOAL.get(s["domain"], s["domain"])
        concepts = evaluate.build_concepts(s["domain"], goal)
        result = cognitive.diagnose(goal, concepts, s["user_text"])
        results.append(
            {
                "id": s["id"],
                "domain": s["domain"],
                "user_text": s["user_text"],
                "expected": s["expected_state"],
                "predicted": result.state,
                "confidence": round(result.confidence, 3),
                "correct": result.state == s["expected_state"],
            }
        )
    return results


def generate_rules(errors: list[dict]) -> list[str]:
    """聚合分析多条错误样本，提炼共性判别原则。失败返回空列表。"""
    if not errors:
        return []
    parts = []
    for i, err in enumerate(errors, 1):
        goal = evaluate.DOMAIN_GOAL.get(err["domain"], err["domain"])
        parts.append(
            f"样本{i}：\n"
            f"  学习者表达：{err['user_text']}\n"
            f"  诊断引擎误判为：{err['predicted']}（{evaluate.STATE_LABEL[err['predicted']]}）\n"
            f"  正确答案应为：{err['expected']}（{evaluate.STATE_LABEL[err['expected']]}）\n"
            f"  学习目标：{goal}"
        )
    user = "\n\n".join(parts)
    data = llm.chat_json(_ANALYZE_SYSTEM, user, temperature=0.3)
    if not data:
        return []
    rules = data.get("rules", [])
    if isinstance(rules, str):
        rules = [rules]
    return [r.strip() for r in rules if r and r.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cognia 诊断自我迭代器")
    parser.add_argument("--rounds", type=int, default=2, help="迭代轮数（默认 2）")
    args = parser.parse_args()

    samples = evaluate.load_samples()
    llm.reset_usage()

    prev_acc: float | None = None

    for rnd in range(1, args.rounds + 1):
        print("=" * 64)
        print(f"第 {rnd} 轮迭代")
        print("=" * 64)
        rules = load_rules()
        print(f"当前经验规则：{len(rules)} 条")

        results = run_eval(samples)
        metrics = evaluate.compute_metrics(results)
        acc = metrics["accuracy"]
        print(f"诊断准确率：{acc * 100:.1f}% （{metrics['correct']}/{metrics['total']}）")

        errors = metrics["errors"]
        if not errors:
            print("✅ 无错误样本，迭代提前收敛。")
            prev_acc = acc
            break

        print(f"错误样本 {len(errors)} 条：")
        for e in errors:
            print(
                f"  - {e['id']}: 期望={evaluate.STATE_LABEL[e['expected']]} "
                f"预测={evaluate.STATE_LABEL[e['predicted']]}"
            )

        # 聚合提炼候选规则（去重）
        candidates = generate_rules(errors)
        to_add = [r for r in candidates if r not in rules]
        if not to_add:
            print("未提炼出新的判别原则，迭代停止。")
            prev_acc = acc
            break

        # 质量门：加入候选规则后复评，只有准确率上升才保留，否则回滚
        save_rules(rules + to_add)
        results2 = run_eval(samples)
        metrics2 = evaluate.compute_metrics(results2)
        acc_after = metrics2["accuracy"]

        if acc_after > acc:
            print(f"质量门通过：{acc * 100:.1f}% -> {acc_after * 100:.1f}%")
            for r in to_add:
                print(f"  + 采纳规则：{r}")
            rules = rules + to_add
            prev_acc = acc_after
        else:
            save_rules(rules)  # 回滚
            print(
                f"质量门未通过（{acc_after * 100:.1f}% <= {acc * 100:.1f}%），"
                "回滚本轮规则，迭代停止。"
            )
            prev_acc = acc
            break

    # 总结
    print()
    print("=" * 64)
    print("迭代总结")
    print("=" * 64)
    if prev_acc is not None:
        print(f"最终诊断准确率：{prev_acc * 100:.1f}%")
    final_rules = load_rules()
    print(f"经验规则累计：{len(final_rules)} 条")
    for r in final_rules:
        print(f"  - {r}")
    usage = llm.get_usage()
    if usage["calls"] > 0:
        print(f"迭代过程 token：{usage['total_tokens']}（{usage['calls']} 次调用）")
    else:
        print("（本次未调用 LLM，可能 AI_ENABLED=0）")
    print("=" * 64)


if __name__ == "__main__":
    main()
