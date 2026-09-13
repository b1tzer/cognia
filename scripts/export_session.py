#!/usr/bin/env python3
"""根据学习记录 ID 导出完整对话记录为 JSON 文件。

用法：
    python3 export_session.py <session_id>
    python3 export_session.py <session_id> --out /path/to/out.json
    python3 export_session.py <session_id> --db /path/to/cognia.db

说明：
- 默认数据库：backend/cognia.db（可用环境变量 COGNIA_DB 覆盖，或 --db 指定）。
- 输出结构与后端 db.get_session(id) 完全一致：
  id / goal / created_at / updated_at / status / stage / messages / cognitive / knowledge。
- messages 中的每条消息完整保留（含 role/content/action/diagnosis/decision/trace 等所有字段，
  其中 trace 即「思考过程」，cognitive/knowledge 为会话快照）。

依赖：仅 Python 标准库（sqlite3 / json / argparse），无需安装任何第三方包。
"""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

DEFAULT_DB = Path(__file__).resolve().parent.parent / "backend" / "cognia.db"


def load_session(db_path: Path, sid: str) -> Optional[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row["id"],
        "goal": row["goal"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "status": row["status"],
        "stage": row["stage"],
        "messages": json.loads(row["messages_json"]),
        "cognitive": json.loads(row["cognitive_json"]) if row["cognitive_json"] else None,
        "knowledge": json.loads(row["knowledge_json"]) if row["knowledge_json"] else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="根据学习记录 ID 导出完整对话记录为 JSON")
    parser.add_argument("session_id", help="学习记录 ID（32 位 hex）")
    parser.add_argument(
        "--db",
        default=os.getenv("COGNIA_DB", str(DEFAULT_DB)),
        help="SQLite 数据库路径（默认 backend/cognia.db，或 $COGNIA_DB）",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        help="输出文件路径（默认 <session_id>.json）",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"错误：数据库文件不存在：{db_path}", file=sys.stderr)
        return 1

    session = load_session(db_path, args.session_id)
    if session is None:
        print(f"错误：未找到 ID 为 {args.session_id} 的学习记录", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else Path(f"{args.session_id}.json")
    out_path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已导出到：{out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
