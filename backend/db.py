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
            stage TEXT NOT NULL DEFAULT 'active',
            messages_json TEXT NOT NULL DEFAULT '[]',
            cognitive_json TEXT,
            knowledge_json TEXT
        )
        """
    )
    # 迁移：老库补 stage 列（clarifying/active/completed 生命周期）
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
    if "stage" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN stage TEXT NOT NULL DEFAULT 'active'")
    # 学习者画像长期记忆层（Phase 3）：跨会话沉淀概念掌握度 / 误解 / 画像
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS learner_profile (
            user_id TEXT PRIMARY KEY,
            prior_level TEXT,
            preferences_json TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS concept_mastery (
            user_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            mastery REAL NOT NULL,
            last_evidence TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, concept_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS misconceptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            concept_id TEXT NOT NULL,
            misconception TEXT NOT NULL,
            confidence REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    # 全局概念库（跨所有历史目标累积的稳定概念，id 为规范化名，跨会话稳定）
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            summary TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # 概念关系（is-a 上下位 / related 横向相关 / prerequisite 前置依赖）
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS concept_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_id TEXT NOT NULL,
            to_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(from_id, to_id, relation_type)
        )
        """
    )
    conn.commit()
    conn.close()


# 学习者画像（长期记忆）默认用户标识。当前无账户系统，固定为全局单用户；
# 未来接入账户系统时，此处替换为真实 user_id 即可，数据模型无需改动。
DEFAULT_USER_ID = "default"

# ---------------------------------------------------------------------------
# 学习者画像（Phase 3 · 跨会话长期记忆）
# ---------------------------------------------------------------------------
def get_learner_profile(user_id: str = DEFAULT_USER_ID) -> Optional[dict]:
    conn = _connect()
    cur = conn.execute("SELECT * FROM learner_profile WHERE user_id = ?", (user_id,))
    r = cur.fetchone()
    conn.close()
    if r is None:
        return None
    return {
        "user_id": r["user_id"],
        "prior_level": r["prior_level"],
        "preferences": json.loads(r["preferences_json"]) if r["preferences_json"] else {},
        "updated_at": r["updated_at"],
    }


def upsert_learner_profile(
    user_id: str = DEFAULT_USER_ID,
    prior_level: Optional[str] = None,
    preferences: Optional[dict] = None,
) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT INTO learner_profile (user_id, prior_level, preferences_json, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            prior_level = excluded.prior_level,
            preferences_json = excluded.preferences_json,
            updated_at = excluded.updated_at
        """,
        (user_id, prior_level, json.dumps(preferences or {}, ensure_ascii=False), _now()),
    )
    conn.commit()
    conn.close()


def upsert_concept_mastery(
    user_id: str,
    concept_id: str,
    mastery: float,
    last_evidence: str = "",
) -> None:
    """写入/更新某概念的掌握度档案（跨 goal 沉淀）。"""
    conn = _connect()
    conn.execute(
        """
        INSERT INTO concept_mastery (user_id, concept_id, mastery, last_evidence, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, concept_id) DO UPDATE SET
            mastery = excluded.mastery,
            last_evidence = excluded.last_evidence,
            updated_at = excluded.updated_at
        """,
        (user_id, concept_id, mastery, last_evidence, _now()),
    )
    conn.commit()
    conn.close()


def get_concept_mastery(user_id: str = DEFAULT_USER_ID) -> dict:
    """返回 {concept_id: 行 dict} 掌握度档案快照；无记录返回空 dict。"""
    conn = _connect()
    cur = conn.execute("SELECT * FROM concept_mastery WHERE user_id = ?", (user_id,))
    rows = cur.fetchall()
    conn.close()
    return {r["concept_id"]: dict(r) for r in rows}


def add_misconception(
    user_id: str,
    concept_id: str,
    misconception: str,
    confidence: float = 0.5,
) -> None:
    """追加一条误解记录（跨 goal 沉淀，去重在 Task 3 处理）。"""
    conn = _connect()
    conn.execute(
        "INSERT INTO misconceptions (user_id, concept_id, misconception, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, concept_id, misconception, confidence, _now()),
    )
    conn.commit()
    conn.close()


def get_misconceptions(
    user_id: str = DEFAULT_USER_ID,
    concept_id: Optional[str] = None,
) -> list[dict]:
    """返回误解记录列表（按创建时间倒序）；可选按 concept_id 过滤。"""
    conn = _connect()
    if concept_id:
        cur = conn.execute(
            "SELECT * FROM misconceptions WHERE user_id = ? AND concept_id = ? ORDER BY created_at DESC",
            (user_id, concept_id),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM misconceptions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_session(
    goal: str,
    knowledge: Optional[dict] = None,
    cognitive: Optional[dict] = None,
    stage: str = "active",
) -> dict:
    sid = uuid.uuid4().hex
    now = _now()
    row = {
        "id": sid,
        "goal": goal,
        "created_at": now,
        "updated_at": now,
        "status": "active",
        "stage": stage,
        "messages": [],
        "cognitive": cognitive,
        "knowledge": knowledge,
    }
    conn = _connect()
    conn.execute(
        "INSERT INTO sessions (id, goal, created_at, updated_at, status, stage, messages_json, cognitive_json, knowledge_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            sid,
            goal,
            now,
            now,
            "active",
            stage,
            "[]",
            json.dumps(cognitive, ensure_ascii=False) if cognitive is not None else None,
            json.dumps(knowledge, ensure_ascii=False) if knowledge is not None else None,
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
        "stage": r["stage"],
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


def update_goal(
    sid: str,
    goal: str,
    knowledge: Optional[dict],
    cognitive: Optional[dict],
    stage: str = "active",
) -> None:
    """更新会话的目标与知识/认知模型，并切换生命周期阶段。

    用于「确认目标」与「中途改目标」两个场景：
    - 澄清阶段（stage=clarifying）：knowledge/cognitive 传 None，仅更新 goal 与阶段
    - 重建阶段（stage=active）：传入新构建的 knowledge/cognitive
    """
    conn = _connect()
    conn.execute(
        "UPDATE sessions SET goal = ?, knowledge_json = ?, cognitive_json = ?, stage = ?, status = ?, updated_at = ? WHERE id = ?",
        (
            goal,
            json.dumps(knowledge, ensure_ascii=False) if knowledge is not None else None,
            json.dumps(cognitive, ensure_ascii=False) if cognitive is not None else None,
            stage,
            "completed" if stage == "completed" else "active",
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


# ---------------------------------------------------------------------------
# 全局概念库（Cognitive Atlas · 跨目标稳定概念）
# ---------------------------------------------------------------------------
def upsert_concept(concept_id: str, name: str, summary: str = "") -> None:
    """写入/更新一个全局概念（id 为规范化名，跨会话稳定）。"""
    conn = _connect()
    conn.execute(
        """
        INSERT INTO concepts (id, name, summary, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            summary = excluded.summary,
            updated_at = excluded.updated_at
        """,
        (concept_id, name, summary, _now(), _now()),
    )
    conn.commit()
    conn.close()

def get_concept(concept_id: str) -> Optional[dict]:
    """按全局概念 id 查单条概念；不存在返回 None。"""
    conn = _connect()
    cur = conn.execute("SELECT * FROM concepts WHERE id = ?", (concept_id,))
    r = cur.fetchone()
    conn.close()
    return dict(r) if r else None

def list_concepts() -> list[dict]:
    """返回全局概念库全部概念（按 id 排序）。"""
    conn = _connect()
    cur = conn.execute("SELECT * FROM concepts ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def upsert_concept_relation(from_id: str, to_id: str, relation_type: str) -> None:
    """写入/更新一条概念关系（幂等：UNIQUE 约束去重，自环忽略）。"""
    if from_id == to_id:
        return
    conn = _connect()
    conn.execute(
        """
        INSERT INTO concept_relations (from_id, to_id, relation_type, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(from_id, to_id, relation_type) DO NOTHING
        """,
        (from_id, to_id, relation_type, _now()),
    )
    conn.commit()
    conn.close()

def list_concept_relations() -> list[dict]:
    """返回全部概念关系（按 id 排序）。"""
    conn = _connect()
    cur = conn.execute("SELECT * FROM concept_relations ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_concept_relations(concept_id: str) -> list[dict]:
    """返回某概念作为 from 或 to 的所有关系。"""
    conn = _connect()
    cur = conn.execute(
        "SELECT * FROM concept_relations WHERE from_id = ? OR to_id = ? ORDER BY id",
        (concept_id, concept_id),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
