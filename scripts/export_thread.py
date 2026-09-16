#!/usr/bin/env python3
"""Cognia 会话导出脚本：从 Supabase PostgreSQL（LangGraph Checkpointer）导出
指定 thread_id 的完整对话 JSON。

数据源权威：`checkpoint_blobs` 表的 messages 通道（msgpack 序列化），
比 HTTP 端点 /threads/{id}/messages 更底层、不依赖后端进程运行。

用法示例：
    # 列出最近的会话（thread_id + title + 时间）
    uv run python scripts/export_thread.py --list

    # 列出会话并限制条数
    uv run python scripts/export_thread.py --list --limit 5

    # 导出指定会话完整对话（pretty JSON 打印到 stdout）
    uv run python scripts/export_thread.py --thread-id <uuid>

    # 导出到文件
    uv run python scripts/export_thread.py --thread-id <uuid> --out /tmp/thread.json

    # 覆盖数据库连接串（默认读项目根 .env 的 LANGGRAPH_DATABASE_URL）
    uv run python scripts/export_thread.py --list --db postgresql://user:pass@host:5432/db

依赖（均为项目已有依赖，无需额外安装）：psycopg / langgraph / dotenv
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

# 项目根目录加入 sys.path，保证脚本从任意 cwd 独立运行
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# 脚本不走应用入口，需自行加载项目根 .env（LANGGRAPH_DATABASE_URL 等）
load_dotenv(ROOT / ".env")

import psycopg
from psycopg.rows import dict_row
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# 与 threads.py 一致的「真实活动」并集：cognia_threads 元数据 ∪ checkpoints，
# 兼容升级前只有 checkpoint、没有 cognia_threads 记录的旧会话。
_LIST_THREADS_SQL = """
SELECT
    COALESCE(t.thread_id, c.thread_id) AS thread_id,
    t.title,
    COALESCE(t.created_at, (c.first_ts)::timestamptz) AS created_at,
    COALESCE((c.latest_ts)::timestamptz, t.updated_at) AS updated_at
FROM cognia_threads t
FULL OUTER JOIN (
    SELECT
        thread_id,
        MIN(checkpoint->>'ts') AS first_ts,
        MAX(checkpoint->>'ts') AS latest_ts
    FROM checkpoints
    WHERE checkpoint_ns = ''
    GROUP BY thread_id
) c ON c.thread_id = t.thread_id
ORDER BY COALESCE((c.latest_ts)::timestamptz, t.updated_at) DESC
LIMIT %s
"""

_GET_META_SQL = """
SELECT thread_id, title, created_at, updated_at
FROM cognia_threads
WHERE thread_id = %s
"""

# checkpoints 的 min/max ts，作为 cognia_threads 缺失时的兜底时间
_GET_TS_FALLBACK_SQL = """
SELECT
    MIN(checkpoint->>'ts') AS first_ts,
    MAX(checkpoint->>'ts') AS latest_ts
FROM checkpoints
WHERE thread_id = %s AND checkpoint_ns = ''
"""

# messages 通道最新快照（version 为左补零递增序号，字典序 DESC 即最新）
_GET_MESSAGES_SQL = """
SELECT blob, type
FROM checkpoint_blobs
WHERE thread_id = %s AND checkpoint_ns = '' AND channel = 'messages'
ORDER BY version DESC
LIMIT 1
"""


def _connect(conn_string: str | None):
    """按连接串建 psycopg 同步连接（autocommit + dict_row）。"""
    url = conn_string or os.getenv("LANGGRAPH_DATABASE_URL")
    if not url:
        raise SystemExit(
            "缺少数据库连接串：请设置 LANGGRAPH_DATABASE_URL（或传 --db）"
        )
    return psycopg.connect(url, row_factory=dict_row, autocommit=True)


def _iso(value) -> str | None:
    """datetime / None / str 统一为 ISO 字符串。"""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def _json_safe(value):
    """递归把 message 里的 content / arguments 转成 JSON 可序列化结构。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _tool_calls_to_list(msg) -> list[dict]:
    """把 AIMessage 的 tool_calls 归一为 [{id, name, arguments}]。"""
    out = []
    for tc in getattr(msg, "tool_calls", None) or []:
        if isinstance(tc, dict):
            out.append({
                "id": tc.get("id"),
                "name": tc.get("name"),
                "arguments": tc.get("args", {}),
            })
        else:
            out.append({
                "id": getattr(tc, "id", None),
                "name": getattr(tc, "name", None),
                "arguments": getattr(tc, "args", {}),
            })
    return out


def _messages_to_dicts(messages: list) -> list[dict]:
    """把 langchain message 对象列表转为可读 dict 列表。

    与 server._history_to_agui 语义对齐：
    - AIMessage 的 reasoning_content（DeepSeek 思考链）拆成独立 role="reasoning"；
    - 其余按 user / assistant / tool / system 分类，tool 消息带 toolCallId。
    """
    out = []
    for raw in messages:
        mtype = getattr(raw, "type", None)
        content = _json_safe(getattr(raw, "content", None))

        if mtype == "ai":
            reasoning = (getattr(raw, "additional_kwargs", None) or {}).get(
                "reasoning_content"
            )
            if isinstance(reasoning, str) and reasoning.strip():
                out.append({
                    "id": f"{getattr(raw, 'id', None)}-reasoning",
                    "role": "reasoning",
                    "content": reasoning,
                })
            out.append({
                "id": getattr(raw, "id", None),
                "role": "assistant",
                "content": content,
                "toolCalls": _tool_calls_to_list(raw),
            })
        elif mtype == "tool":
            out.append({
                "id": getattr(raw, "id", None),
                "role": "tool",
                "content": content,
                "toolCallId": getattr(raw, "tool_call_id", None),
            })
        elif mtype == "human":
            out.append({
                "id": getattr(raw, "id", None),
                "role": "user",
                "content": content,
            })
        elif mtype == "system":
            out.append({
                "id": getattr(raw, "id", None),
                "role": "system",
                "content": content,
            })
        else:
            out.append({
                "id": getattr(raw, "id", None),
                "role": mtype or "unknown",
                "content": content,
            })
    return out


def list_threads(conn, limit: int) -> list[dict]:
    """列出会话（thread_id + title + 时间），按最近活动时间降序。"""
    with conn.cursor() as cur:
        cur.execute(_LIST_THREADS_SQL, (limit,))
        rows = cur.fetchall()
    return [
        {
            "thread_id": r["thread_id"],
            "title": r["title"],
            "created_at": _iso(r["created_at"]),
            "updated_at": _iso(r["updated_at"]),
        }
        for r in rows
    ]


def export_thread(conn, thread_id: str) -> dict:
    """导出单个 thread 的元数据 + 完整对话消息。"""
    # 1) 元数据（cognia_threads 提供 title 与 created_at 兜底）
    meta = None
    with conn.cursor() as cur:
        cur.execute(_GET_META_SQL, (thread_id,))
        row = cur.fetchone()
        if row:
            meta = row

    # 2) checkpoints 的 min/max ts：真实对话活动时间（与 list_threads 口径一致）
    first_ts = latest_ts = None
    with conn.cursor() as cur:
        cur.execute(_GET_TS_FALLBACK_SQL, (thread_id,))
        ts = cur.fetchone()
        if ts:
            first_ts = _iso(ts["first_ts"])
            latest_ts = _iso(ts["latest_ts"])

    title = meta["title"] if meta else None
    created_at = _iso(meta["created_at"]) if meta else first_ts
    updated_at = latest_ts or (_iso(meta["updated_at"]) if meta else None)

    # 3) 最新 messages 快照
    messages = []
    with conn.cursor() as cur:
        cur.execute(_GET_MESSAGES_SQL, (thread_id,))
        row = cur.fetchone()
        if row:
            ser = JsonPlusSerializer()
            raw_messages = ser.loads_typed((row["type"], bytes(row["blob"])))
            if isinstance(raw_messages, list):
                messages = _messages_to_dicts(raw_messages)

    return {
        "thread_id": thread_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "message_count": len(messages),
        "messages": messages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 Supabase Postgres（LangGraph Checkpointer）导出会话对话 JSON"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出最近的会话（thread_id + title + 时间）",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="要导出的会话 thread_id（UUID）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="配合 --list：最多列出的会话数（默认 20）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="可选：导出结果 JSON 落盘路径（默认打印到 stdout）",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="可选：覆盖数据库连接串（默认读 .env 的 LANGGRAPH_DATABASE_URL）",
    )
    args = parser.parse_args()

    conn = _connect(args.db)
    try:
        if args.thread_id:
            payload = export_thread(conn, args.thread_id)
        else:
            payload = list_threads(conn, args.limit)

        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
            print(f"已写入: {args.out}")
        else:
            print(text)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
