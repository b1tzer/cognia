#!/usr/bin/env python3
"""Cognia 历史会话标题批量重新生成脚本。

把「旧版截断首条消息 30 字」的历史标题，用新的 AI 精炼逻辑重新生成：
- 读每个会话的首条用户消息（checkpoint_blobs 的 messages 通道，权威来源）；
- 调 planner 模型生成 ≤15 字精炼标题；
- 更新 cognia_threads.title。

默认 dry-run：只打印「旧标题 → 新标题」对比，不写库；加 --apply 才实际更新。
AI 生成失败或读不到首条消息的会话跳过，不覆盖旧标题。

用法示例：
    # 预览前 20 个会话的新旧标题对比（不写库）
    uv run python scripts/regenerate_titles.py --limit 20

    # 实际写库（全部会话）
    uv run python scripts/regenerate_titles.py --apply

    # 覆盖数据库连接串（默认读 .env 的 LANGGRAPH_DATABASE_URL）
    uv run python scripts/regenerate_titles.py --apply --db postgresql://user:pass@host:5432/db
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from cognia import models

# 与 cognia/routers/threads.py 的 _TITLE_SYSTEM_PROMPT 保持一致
TITLE_SYSTEM_PROMPT = (
    "你是会话标题生成助手。根据用户的第一条消息，提炼一个不超过15个字的"
    "精炼中文标题，直接概括用户的核心意图或问题。只输出标题本身，"
    "不要加引号、句号等标点，不要输出任何解释。"
)

# 与 threads.py 一致的「真实活动」并集：cognia_threads ∪ checkpoints
_LIST_THREADS_SQL = """
SELECT
    COALESCE(t.thread_id, c.thread_id) AS thread_id,
    t.title
FROM cognia_threads t
FULL OUTER JOIN (
    SELECT thread_id
    FROM checkpoints
    WHERE checkpoint_ns = ''
    GROUP BY thread_id
) c ON c.thread_id = t.thread_id
ORDER BY COALESCE(t.thread_id, c.thread_id)
"""

# 最新 messages 快照（version 左补零递增，字典序 DESC 即最新）
_GET_MESSAGES_SQL = """
SELECT blob, type
FROM checkpoint_blobs
WHERE thread_id = %s AND checkpoint_ns = '' AND channel = 'messages'
ORDER BY version DESC
LIMIT 1
"""

_UPDATE_TITLE_SQL = """
UPDATE cognia_threads
SET title = %s, updated_at = now()
WHERE thread_id = %s
"""


def _connect(conn_string: str | None):
    url = conn_string or os.getenv("LANGGRAPH_DATABASE_URL")
    if not url:
        raise SystemExit("缺少数据库连接串：请设置 LANGGRAPH_DATABASE_URL（或传 --db）")
    return psycopg.connect(url, row_factory=dict_row, autocommit=True)


def _first_user_text(messages) -> str | None:
    """从反序列化的 langchain 消息列表取首条用户文本，用于生成标题。"""
    for msg in messages or []:
        if getattr(msg, "type", None) != "human":
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        return text
    return None


def _load_first_user_text(conn, thread_id: str) -> str | None:
    """读某会话首条用户消息文本；无 checkpoint / 无用户消息返回 None。"""
    with conn.cursor() as cur:
        cur.execute(_GET_MESSAGES_SQL, (thread_id,))
        row = cur.fetchone()
    if not row:
        return None
    ser = JsonPlusSerializer()
    raw = ser.loads_typed((row["type"], bytes(row["blob"])))
    if isinstance(raw, list):
        return _first_user_text(raw)
    return None


def _generate_title(model, text: str) -> str | None:
    """调 LLM 生成精炼标题；失败 / 空结果返回 None（调用方跳过）。"""
    try:
        resp = model.invoke([
            ("system", TITLE_SYSTEM_PROMPT),
            ("human", text),
        ])
    except Exception as exc:
        print(f"      [警告] LLM 调用异常：{exc}")
        return None
    title = (resp.content or "").strip()
    if not title:
        return None
    return title[:15] + ("…" if len(title) > 15 else "")


def _iter_threads(conn, limit: int):
    with conn.cursor() as cur:
        cur.execute(_LIST_THREADS_SQL)
        rows = cur.fetchall()
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser(description="历史会话标题 AI 精炼重新生成")
    parser.add_argument("--apply", action="store_true", help="实际写库（默认 dry-run 只打印）")
    parser.add_argument("--limit", type=int, default=None, help="最多处理的会话数（默认全部）")
    parser.add_argument("--db", default=None, help="覆盖数据库连接串（默认读 .env）")
    args = parser.parse_args()

    # 行缓冲：非 tty 环境下逐行实时输出，避免进度攒在缓冲区里看不到
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    conn = _connect(args.db)
    model = models.get_planner_model()
    updated = skipped = 0

    try:
        rows = _iter_threads(conn, args.limit)
        print(f"共 {len(rows)} 个会话待处理（mode={'apply' if args.apply else 'dry-run'}）\n")
        for r in rows:
            thread_id = r["thread_id"]
            old_title = r["title"]
            text = _load_first_user_text(conn, thread_id)
            if not text:
                skipped += 1
                print(f"[跳过] {thread_id}: 无首条用户消息（title={old_title!r}）")
                continue
            new_title = _generate_title(model, text)
            if not new_title:
                skipped += 1
                print(f"[跳过] {thread_id}: AI 生成失败（title={old_title!r}）")
                continue
            if args.apply:
                with conn.cursor() as cur:
                    cur.execute(_UPDATE_TITLE_SQL, (new_title, thread_id))
            updated += 1
            print(f"[{'更新' if args.apply else '预览'}] {thread_id}\n    旧: {old_title!r}\n    新: {new_title!r}")
    finally:
        conn.close()

    print(f"\n完成：更新/预览 {updated} 个，跳过 {skipped} 个"
          + ("" if args.apply else "（dry-run 未写库，加 --apply 实际执行）"))


if __name__ == "__main__":
    main()
