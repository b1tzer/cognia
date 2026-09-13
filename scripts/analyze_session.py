#!/usr/bin/env python3
"""将学习记录 JSON 解析为结构化的对话分析报告。

这是分析单条学习记录最常用的动作：把 export_session.py 导出的完整 JSON
（messages / cognitive / knowledge / trace 全字段）压平成一份人类可读的分析报告，
覆盖四个维度：
  1. 会话概要（目标 / 状态 / 消息数 / 时长）
  2. 概念掌握度快照（cognitive.concepts 的 mastery / state / evidence_count）
  3. 逐轮对话轨迹（role / action / diagnosis / decision / content 摘要）
  4. token 成本（trace[].usage 的 prompt/completion 汇总与模型分布）

用法：
    python3 analyze_session.py <session_id>
    python3 analyze_session.py <session_id> --db /path/to/cognia.db
    python3 analyze_session.py <session_id> --out /path/to/report.txt
    python3 analyze_session.py --file /path/to/session.json

说明：
- 默认直接按 session_id 从数据库读取（复用同目录 export_session.load_session），
  也可用 --file 传入已导出的 JSON 文件离线分析。
- 报告默认打印到 stdout，可用 --out 写入文件。

依赖：仅 Python 标准库，无需第三方包。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# 允许直接复用同目录 export_session.py 的 load_session
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_session import load_session, DEFAULT_DB  # noqa: E402


def fmt_snippet(text: str, limit: int = 120) -> str:
    """压缩长文本为单行摘要，避免报告被长内容撑爆。"""
    if not text:
        return ""
    one_line = " ".join(str(text).split())
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + "…"


def section(title: str) -> str:
    return f"\n{'=' * 80}\n{title}\n{'=' * 80}"


def render_overview(s: dict) -> str:
    lines = [section("一、会话概要")]
    lines.append(f"目标 goal       : {s.get('goal')}")
    lines.append(f"状态 status     : {s.get('status')}")
    lines.append(f"阶段 stage      : {s.get('stage')}")
    lines.append(f"创建时间        : {s.get('created_at')}")
    lines.append(f"更新时间        : {s.get('updated_at')}")
    msgs = s.get("messages") or []
    lines.append(f"消息总数        : {len(msgs)}")
    return "\n".join(lines)


def render_concepts(s: dict) -> str:
    concepts = (s.get("cognitive") or {}).get("concepts") or []
    lines = [section(f"二、概念掌握度快照（{len(concepts)} 个）")]
    if not concepts:
        lines.append("（无概念快照）")
        return "\n".join(lines)
    for c in concepts:
        name = c.get("concept_name") or c.get("name") or "?"
        mastery = c.get("mastery")
        state = c.get("state")
        ev = c.get("evidence_count")
        mastery_s = f"{mastery:.4f}" if isinstance(mastery, (int, float)) else str(mastery)
        lines.append(
            f"  {name:<22} mastery={mastery_s:<8} state={state:<12} evidence={ev}"
        )
    return "\n".join(lines)


def render_dialogue(s: dict) -> str:
    msgs = s.get("messages") or []
    lines = [section(f"三、逐轮对话轨迹（{len(msgs)} 条消息）")]
    for i, m in enumerate(msgs):
        role = m.get("role")
        action = m.get("action")
        lines.append(f"\n[{i:02d}] role={role} action={action}")

        diag = m.get("diagnosis")
        if diag:
            lines.append(
                f"     诊断: state={diag.get('state')} conf={diag.get('confidence')} "
                f"concepts={diag.get('concept_ids')}"
            )
            if diag.get("missing"):
                lines.append(f"           缺失: {diag['missing']}")
            if diag.get("misconception"):
                lines.append(f"           误解: {diag['misconception']}")

        content = m.get("content")
        if content:
            lines.append(f"     内容: {fmt_snippet(content)}")

        dec = m.get("decision")
        if dec:
            reasons = dec.get("reasons") or {}
            crit = reasons.get("criterion_used") or reasons.get("criterion")
            intent = reasons.get("pedagogical_intent")
            lines.append(
                f"     决策: chosen={dec.get('chosen_action')} criterion={crit}"
            )
            if intent:
                lines.append(f"           意图: {fmt_snippet(intent, 80)}")
    return "\n".join(lines)


def render_tokens(s: dict) -> str:
    msgs = s.get("messages") or []
    lines = [section("四、token 成本")]
    total_in = total_out = 0
    models: dict = {}
    trace_count = 0
    for m in msgs:
        for t in m.get("trace") or []:
            trace_count += 1
            usage = t.get("usage") or {}
            total_in += usage.get("prompt_tokens", 0) or 0
            total_out += usage.get("completion_tokens", 0) or 0
            model = t.get("model") or "unknown"
            models[model] = models.get(model, 0) + 1
    lines.append(f"trace 条数       : {trace_count}")
    lines.append(f"模型调用分布    : {models if models else '（无）'}")
    lines.append(f"prompt_tokens   : {total_in}")
    lines.append(f"completion_tokens: {total_out}")
    lines.append(f"合计 tokens     : {total_in + total_out}")
    return "\n".join(lines)


def render_report(s: dict) -> str:
    parts = [
        render_overview(s),
        render_concepts(s),
        render_dialogue(s),
        render_tokens(s),
    ]
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将学习记录 JSON 解析为结构化分析报告"
    )
    parser.add_argument("session_id", nargs="?", help="学习记录 ID（32 位 hex）")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help=f"SQLite 数据库路径（默认 {DEFAULT_DB}，或 $COGNIA_DB）",
    )
    parser.add_argument("--file", help="已导出的 session JSON 文件路径（与 session_id 二选一）")
    parser.add_argument("--out", "-o", help="报告输出文件（默认打印到 stdout）")
    args = parser.parse_args()

    session: Optional[dict]
    if args.file:
        fpath = Path(args.file)
        if not fpath.exists():
            print(f"错误：JSON 文件不存在：{fpath}", file=sys.stderr)
            return 1
        session = json.loads(fpath.read_text(encoding="utf-8"))
    else:
        if not args.session_id:
            print("错误：需提供 session_id 或 --file 之一", file=sys.stderr)
            return 1
        db_path = Path(args.db)
        if not db_path.exists():
            print(f"错误：数据库文件不存在：{db_path}", file=sys.stderr)
            return 1
        session = load_session(db_path, args.session_id)
        if session is None:
            print(f"错误：未找到 ID 为 {args.session_id} 的学习记录", file=sys.stderr)
            return 1

    report = render_report(session)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"报告已写入：{Path(args.out).resolve()}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
