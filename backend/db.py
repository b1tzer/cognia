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
