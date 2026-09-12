"""对话栈随 cognitive 持久化的回归测试（Phase 1 · Task 3 · 验收标准 4）。

验证：dialogue 轨迹在 chat 流程结束后随 cognitive 一起落库，重新读取不丢失。
"""
from __future__ import annotations

import os
import tempfile
import unittest

import config

config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "dialogue_persist_test.db")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import db  # noqa: E402


class TestDialoguePersistence(unittest.TestCase):
    def test_dialogue_persisted_after_chat(self):
        with TestClient(main.app) as client:
            sid = client.post("/api/sessions", json={"goal": "理解 HTTP"}).json()["id"]
            client.post(f"/api/sessions/{sid}/chat", json={"content": "HTTP 是无状态协议，客户端发请求服务器响应"})
            # 重新从 db 读取，焦点概念应有 1 条对话轨迹
            s = db.get_session(sid)
            stacks = [c.get("dialogue", []) for c in s["cognitive"]["concepts"]]
            non_empty = [x for x in stacks if x]
            self.assertEqual(len(non_empty), 1)
            self.assertEqual(non_empty[0][0]["user_text"], "HTTP 是无状态协议，客户端发请求服务器响应")


if __name__ == "__main__":
    unittest.main()
