"""会话级对话消息持久化的回归测试。

验证：chat 流程结束后，对话以会话级 messages 落库（需求 #71 需求2 废弃了
概念级 dialogue 栈，改用 sessions.messages 作为权威对话原文），重新读取不丢失。
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
            # 重新从 db 读取，会话级 messages 应持久化（开场白 + 用户 + 助手）
            s = db.get_session(sid)
            messages = s["messages"]
            roles = [m["role"] for m in messages]
            self.assertIn("user", roles)
            self.assertIn("assistant", roles)
            # 用户消息内容不丢失
            user_msgs = [m for m in messages if m["role"] == "user"]
            self.assertEqual(user_msgs[0]["content"], "HTTP 是无状态协议，客户端发请求服务器响应")


if __name__ == "__main__":
    unittest.main()
