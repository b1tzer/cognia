"""SQLite 持久化层。

会话、消息、知识模型、认知模型均以 JSON 序列化存储，
保证长会话 / 跨进程重启后学习状态不丢失。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            goal TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            messages_json TEXT NOT NULL DEFAULT '[]',
            cognitive_json TEXT,
            knowledge_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def create_session(goal: str, knowledge: dict, cognitive: dict) -> dict:
    sid = uuid.uuid4().hex
    now = _now()
    row = {
        "id": sid,
        "goal": goal,
        "created_at": now,
        "updated_at": now,
        "status": "active",
        "messages": [],
        "cognitive": cognitive,
        "knowledge": knowledge,
    }
    conn = _connect()
    conn.execute(
        "INSERT INTO sessions (id, goal, created_at, updated_at, status, messages_json, cognitive_json, knowledge_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            sid,
            goal,
            now,
            now,
            "active",
            "[]",
            json.dumps(cognitive, ensure_ascii=False),
            json.dumps(knowledge, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()
    return row


def get_session(sid: str) -> Optional[dict]:
    conn = _connect()
    cur = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,))
    r = cur.fetchone()
    conn.close()
    if r is None:
        return None
    return {
        "id": r["id"],
        "goal": r["goal"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "status": r["status"],
        "messages": json.loads(r["messages_json"]),
        "cognitive": json.loads(r["cognitive_json"]) if r["cognitive_json"] else None,
        "knowledge": json.loads(r["knowledge_json"]) if r["knowledge_json"] else None,
    }


def append_messages(sid: str, messages: list[dict]) -> None:
    conn = _connect()
    cur = conn.execute("SELECT messages_json FROM sessions WHERE id = ?", (sid,))
    r = cur.fetchone()
    existing = json.loads(r["messages_json"]) if r else []
    existing.extend(messages)
    conn.execute(
        "UPDATE sessions SET messages_json = ?, updated_at = ? WHERE id = ?",
        (json.dumps(existing, ensure_ascii=False), _now(), sid),
    )
    conn.commit()
    conn.close()


def append_user_message(sid: str, content: str) -> None:
    """立即追加用户消息（diagnosis 暂空），避免流式回复期间用户刷新导致消息丢失。"""
    append_messages(sid, [{"role": "user", "content": content, "action": None, "diagnosis": None}])


def update_last_user_diagnosis(sid: str, diagnosis: dict) -> None:
    """回填最后一条用户消息的 diagnosis（流程走完后补全诊断结果）。"""
    conn = _connect()
    cur = conn.execute("SELECT messages_json FROM sessions WHERE id = ?", (sid,))
    r = cur.fetchone()
    if r is None:
        conn.close()
        return
    messages = json.loads(r["messages_json"])
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages[i]["diagnosis"] = diagnosis
            break
    conn.execute(
        "UPDATE sessions SET messages_json = ?, updated_at = ? WHERE id = ?",
        (json.dumps(messages, ensure_ascii=False), _now(), sid),
    )
    conn.commit()
    conn.close()


def update_session(sid: str, cognitive: dict, knowledge: dict, status: str = "active") -> None:
    conn = _connect()
    conn.execute(
        "UPDATE sessions SET cognitive_json = ?, knowledge_json = ?, status = ?, updated_at = ? WHERE id = ?",
        (
            json.dumps(cognitive, ensure_ascii=False),
            json.dumps(knowledge, ensure_ascii=False),
            status,
            _now(),
            sid,
        ),
    )
    conn.commit()
    conn.close()


def list_sessions() -> list[dict]:
    conn = _connect()
    cur = conn.execute("SELECT id, goal, created_at, updated_at, status FROM sessions ORDER BY updated_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def recent_sessions(limit: int = 50) -> list[dict]:
    """按最近更新倒序返回最多 limit 条完整会话（含 messages/knowledge/cognitive）。

    供在线 Prompt 优化器提取真实对话配对使用。
    """
    conn = _connect()
    cur = conn.execute(
        "SELECT id FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
    )
    ids = [r["id"] for r in cur.fetchall()]
    conn.close()
    sessions = []
    for sid in ids:
        s = get_session(sid)
        if s is not None:
            sessions.append(s)
    return sessions


def delete_session(sid: str) -> bool:
    """删除整个学习目标（会话）。返回是否删除成功。"""
    conn = _connect()
    cur = conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    conn.commit()
    conn.close()
    return cur.rowcount > 0


def delete_message(sid: str, index: int) -> Optional[list[dict]]:
    """删除会话中指定索引的消息，返回删除后的消息列表；失败返回 None。"""
    conn = _connect()
    cur = conn.execute("SELECT messages_json FROM sessions WHERE id = ?", (sid,))
    r = cur.fetchone()
    if r is None:
        conn.close()
        return None
    messages = json.loads(r["messages_json"])
    if index < 0 or index >= len(messages):
        conn.close()
        return None
    messages.pop(index)
    conn.execute(
        "UPDATE sessions SET messages_json = ?, updated_at = ? WHERE id = ?",
        (json.dumps(messages, ensure_ascii=False), _now(), sid),
    )
    conn.commit()
    conn.close()
    return messages


def update_message(sid: str, index: int, content: str) -> Optional[list[dict]]:
    """修改会话中指定索引消息的 content，返回修改后的消息列表；失败返回 None。"""
    conn = _connect()
    cur = conn.execute("SELECT messages_json FROM sessions WHERE id = ?", (sid,))
    r = cur.fetchone()
    if r is None:
        conn.close()
        return None
    messages = json.loads(r["messages_json"])
    if index < 0 or index >= len(messages):
        conn.close()
        return None
    messages[index]["content"] = content
    conn.execute(
        "UPDATE sessions SET messages_json = ?, updated_at = ? WHERE id = ?",
        (json.dumps(messages, ensure_ascii=False), _now(), sid),
    )
    conn.commit()
    conn.close()
    return messages
