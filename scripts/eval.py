#!/usr/bin/env python3
"""Cognia 诊断评估 harness（任务⑧）。

跑金标集 → 复用生产 run_diagnosis 诊断 → 对比专家标注 → 输出多维指标：
整体准确率、分状态准确率（recall）、混淆矩阵（5×5）。

用法示例：
    # 开发集全量评估
    python scripts/eval.py --data data/golden/dev_set.json

    # 盲测集评估 + 结果落盘 JSON
    python scripts/eval.py --data data/golden/test_set.json --out eval_result.json

    # 冒烟测试：只跑前 N 条（验证脚本可跑通，不烧 token）
    python scripts/eval.py --data data/golden/dev_set.json --limit 3

    # 临时切换诊断模型（覆盖 DIAGNOSER_MODEL）
    python scripts/eval.py --data data/golden/dev_set.json --model deepseek-v4-pro-external

依赖环境变量（与生产一致，脚本会自动加载项目根 .env）：
    LLM_API_BASE / LLM_API_KEY / DIAGNOSER_MODEL
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# 项目根目录加入 sys.path，保证脚本从任意 cwd 独立运行都能 import cognia 包
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# 脚本不走应用入口，需自行加载项目根 .env（LLM_API_BASE / LLM_API_KEY 等）
load_dotenv(ROOT / ".env")

from cognia.learning_engine import run_diagnosis
from cognia.models import get_diagnoser_model
from cognia.schemas import CognitiveState, KnowledgePoint

# 五态固定顺序（混淆矩阵行/列、分状态指标都按此排序，保证输出稳定）
STATES = [s.value for s in CognitiveState]


def load_samples(path: str) -> list[dict]:
    """加载金标集 JSON，返回样本列表。"""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"金标集文件不存在：{p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise SystemExit(f"金标集 {p} 缺少 samples 列表或列表为空")
    return samples


def _diagnose_one(sample: dict, diagnoser) -> dict:
    """单样本诊断，异常不中断整体评估。"""
    item = {
        "id": sample.get("id"),
        "truth": sample.get("expected_state"),
        "pred": None,
        "confidence": None,
        "ok": None,
        "error": None,
    }
    try:
        point = KnowledgePoint(**sample["point"])
        diagnosis = run_diagnosis(
            diagnoser,
            point,
            sample["question"],
            sample["user_answer"],
        )
        item["pred"] = diagnosis.state.value
        item["confidence"] = diagnosis.confidence.value
        item["ok"] = diagnosis.state.value == sample.get("expected_state")
    except Exception as exc:  # noqa: BLE001 —— LLM 调用失败不中断整体评估
        item["error"] = f"{type(exc).__name__}: {exc}"
    return item


def _log_progress(done: int, total: int, item: dict) -> None:
    """逐样本进度打印（flush 保证即时可见，避免长任务看起来卡住）。"""
    mark = "OK " if item["ok"] else ("ERR" if item["error"] is not None else "MISS")
    status = item["error"] or f"{item['pred']} vs {item['truth']}"
    print(f"[{done}/{total}] {mark} {item['id']}: {status}", flush=True)


def run_eval(samples: list[dict], diagnoser, workers: int = 1) -> list[dict]:
    """逐样本跑生产诊断逻辑，返回每样本的预测与真值对照。

    workers > 1 时用线程池并发（LLM 调用为 IO 密集，线程并发可显著加速），
    结果仍保持样本原始顺序。
    """
    total = len(samples)
    results = []
    if workers <= 1 or total <= 1:
        for s in samples:
            item = _diagnose_one(s, diagnoser)
            results.append(item)
            _log_progress(len(results), total, item)
        return results

    ordered = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_diagnose_one, s, diagnoser): i for i, s in enumerate(samples)}
        done = 0
        for fut in as_completed(futures):
            item = fut.result()
            ordered[futures[fut]] = item
            done += 1
            _log_progress(done, total, item)
    return ordered


def summarize(results: list[dict]) -> dict:
    """汇总多维指标：总体准确率 + 分状态准确率 + 混淆矩阵。"""
    succeeded = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]

    # 总体准确率：仅在成功诊断的样本上计算（失败样本不算命中也不算分母）
    correct = sum(1 for r in succeeded if r["ok"])
    total = len(succeeded)
    overall = correct / total if total else 0.0

    # 分状态准确率（recall：某真值状态下，被正确诊断的比例）
    per_state = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in succeeded:
        per_state[r["truth"]]["total"] += 1
        if r["ok"]:
            per_state[r["truth"]]["correct"] += 1

    # 混淆矩阵：行=真值（ground truth），列=预测（prediction）
    matrix = {truth: {pred: 0 for pred in STATES} for truth in STATES}
    for r in succeeded:
        matrix[r["truth"]][r["pred"]] += 1

    return {
        "total": len(results),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "overall_accuracy": overall,
        "correct": correct,
        "per_state_accuracy": {
            s: {
                "correct": per_state[s]["correct"],
                "total": per_state[s]["total"],
                "accuracy": (
                    per_state[s]["correct"] / per_state[s]["total"]
                    if per_state[s]["total"]
                    else None
                ),
            }
            for s in STATES
        },
        "confusion_matrix": matrix,
    }


def render_report(data_path: str, model_id: str, summary: dict, results: list[dict]) -> str:
    """渲染可读文本报告。"""
    lines = []
    lines.append("=== Cognia 诊断评估报告 ===")
    lines.append(f"数据文件 : {data_path}")
    lines.append(f"诊断模型 : {model_id}")
    lines.append(f"样本总数 : {summary['total']}")
    lines.append(f"成功诊断 : {summary['succeeded']}")
    lines.append(f"失败样本 : {summary['failed']}")
    if summary["failed"]:
        for r in results:
            if r["error"]:
                lines.append(f"  - {r['id']}: {r['error']}")
    lines.append("")
    lines.append(
        f"总体准确率 : {summary['overall_accuracy']:.3f} "
        f"({summary['correct']}/{summary['succeeded']})"
    )
    lines.append("")
    lines.append("分状态准确率（recall）:")
    for s in STATES:
        st = summary["per_state_accuracy"][s]
        acc = f"{st['accuracy']:.3f}" if st["accuracy"] is not None else "  - "
        lines.append(f"  {s:<13} {st['correct']}/{st['total']}  = {acc}")
    lines.append("")
    lines.append("混淆矩阵（行=真值, 列=预测）:")
    header = "  " + "".join(f"{s[:11]:>12}" for s in STATES)
    lines.append(header)
    for truth in STATES:
        row = "  " + f"{truth:<12}"
        for pred in STATES:
            row += f"{summary['confusion_matrix'][truth][pred]:>12}"
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cognia 诊断评估 harness：跑金标集，输出准确率与混淆矩阵"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="金标集 JSON 路径（如 data/golden/dev_set.json）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="可选：评估结果 JSON 落盘路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="可选：只跑前 N 条样本（冒烟测试用，不烧 token）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="可选：并发诊断线程数（默认 1 串行；LLM 为 IO 密集，可设 4-8 加速）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="可选：覆盖 DIAGNOSER_MODEL 环境变量",
    )
    args = parser.parse_args()

    samples = load_samples(args.data)
    if args.limit is not None:
        samples = samples[: args.limit]

    if args.model:
        os.environ["DIAGNOSER_MODEL"] = args.model

    diagnoser = get_diagnoser_model()
    model_id = os.getenv("DIAGNOSER_MODEL", "deepseek-v4-flash")

    results = run_eval(samples, diagnoser, workers=args.workers)
    summary = summarize(results)

    print(render_report(args.data, model_id, summary, results))

    if args.out:
        payload = {
            "data": args.data,
            "model": model_id,
            "summary": summary,
            "details": results,
        }
        Path(args.out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n评估结果已写入: {args.out}")


if __name__ == "__main__":
    main()
