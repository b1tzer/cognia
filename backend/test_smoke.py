"""端到端 smoke test：验证后端闭环（create_session → chat）能真正跑通。

用 FastAPI TestClient + 临时数据库 + AI_ENABLED=0（强制启发式/模板模式），
保证测试不依赖外部模型、不污染真实 cognia.db。
"""
from __future__ import annotations

import os
import tempfile
import unittest

import config

# 必须在 import main/db 之前设置，避免污染真实数据库、强制离线规则路径
config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "smoke_test.db")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402


class TestSmoke(unittest.TestCase):
    def test_full_loop(self):
        with TestClient(main.app) as client:
            # 1. 创建会话
            r = client.post("/api/sessions", json={"goal": "理解 HTTP 协议"})
            self.assertEqual(r.status_code, 200)
            session = r.json()
            sid = session["id"]
            self.assertEqual(len(session["messages"]), 1)  # 开场白
            self.assertIn("concepts", session["cognitive"])

            # 2. 发送消息，验证闭环返回结构完整
            r2 = client.post(
                f"/api/sessions/{sid}/chat",
                json={"content": "HTTP 是无状态的请求-响应协议"},
            )
            self.assertEqual(r2.status_code, 200)
            body = r2.json()
            self.assertIn(body["action"], ("probe", "explain", "correct", "backtrack", "advance"))
            self.assertTrue(body["reply"])
            self.assertIn("state", body["diagnosis"])
            # 认知模型每个概念都应有 consecutive_failures 字段（数据完整性）
            for c in body["cognitive"]["concepts"]:
                self.assertIn("consecutive_failures", c)
                self.assertIn("evidence_count", c)

            # 3. 能取回会话，消息已持久化
            r3 = client.get(f"/api/sessions/{sid}")
            self.assertEqual(r3.status_code, 200)
            self.assertGreaterEqual(len(r3.json()["messages"]), 3)  # 开场白 + 用户 + AI

    def test_empty_goal_rejected(self):
        with TestClient(main.app) as client:
            r = client.post("/api/sessions", json={"goal": "   "})
            self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
