"""「刷新丢消息」修复的回归测试。

背景：流式改造后，用户消息与 AI 回复原先都在流式结束才一次性落库，
导致用户在流式打字过程中刷新页面时，用户消息从数据库消失。

修复后不变量：
1. 用户消息在 _process_turn 之前就立即落库（diagnosis 暂空）。
2. _process_turn / 流式 / _persist 期间，用户消息已持久化，刷新可查回。
3. AI 回复在流式结束后追加，且用户消息不会重复（messages 数量正确）。

强制 AI_ENABLED=False 走确定性路径，不依赖外部模型。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import config

config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "msg_persist_test.db")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import db  # noqa: E402


class TestDbHelpers(unittest.TestCase):
    def test_append_user_message(self):
        with TestClient(main.app) as client:
            sid = client.post("/api/sessions", json={"goal": "理解AQS"}).json()["id"]
            db.append_user_message(sid, "我的理解")
            s = db.get_session(sid)
            last = s["messages"][-1]
            self.assertEqual(last["role"], "user")
            self.assertEqual(last["content"], "我的理解")
            self.assertIsNone(last["diagnosis"])

    def test_update_last_user_diagnosis(self):
        with TestClient(main.app) as client:
            sid = client.post("/api/sessions", json={"goal": "理解AQS"}).json()["id"]
            db.append_user_message(sid, "我的理解")
            db.update_last_user_diagnosis(sid, {"state": "partial", "confidence": 0.5})
            s = db.get_session(sid)
            last = s["messages"][-1]
            self.assertEqual(last["diagnosis"]["state"], "partial")


class TestChatPersistence(unittest.TestCase):
    def test_user_message_persisted_before_process_turn(self):
        """核心不变量：_persist_user_message 必须先于 _process_turn 执行。"""
        calls: list[str] = []
        orig_process = main._process_turn
        orig_persist_user = main._persist_user_message

        def spy_process(s, content):
            calls.append("process_turn")
            return orig_process(s, content)

        def spy_persist_user(sid, content):
            calls.append("persist_user")
            return orig_persist_user(sid, content)

        with mock.patch.object(main, "_process_turn", side_effect=spy_process), \
             mock.patch.object(main, "_persist_user_message", side_effect=spy_persist_user):
            with TestClient(main.app) as client:
                sid = client.post("/api/sessions", json={"goal": "理解AQS"}).json()["id"]
                client.post(f"/api/sessions/{sid}/chat", json={"content": "并发基础"})

        self.assertEqual(calls[0], "persist_user")
        self.assertEqual(calls[1], "process_turn")

    def test_chat_messages_not_duplicated(self):
        """改造后用户消息只落库一次：intro + user + ai = 3 条。"""
        with TestClient(main.app) as client:
            sid = client.post("/api/sessions", json={"goal": "理解AQS"}).json()["id"]
            r = client.post(f"/api/sessions/{sid}/chat", json={"content": "并发基础是线程安全协调的基础"})
            self.assertEqual(r.status_code, 200)
            s = db.get_session(sid)
            # 开场白(assistant) + 用户消息 + AI 回复
            self.assertEqual(len(s["messages"]), 3)
            roles = [m["role"] for m in s["messages"]]
            self.assertEqual(roles, ["assistant", "user", "assistant"])
            # 用户消息 content 正确且 diagnosis 已回填
            user_msg = s["messages"][1]
            self.assertEqual(user_msg["content"], "并发基础是线程安全协调的基础")
            self.assertIsNotNone(user_msg["diagnosis"])
            self.assertIn("state", user_msg["diagnosis"])

    def test_assistant_message_does_not_duplicate_diagnosis(self):
        """诊断结果只应挂在 user 消息，assistant 消息不应冗余存一份（问题 7 去重）。"""
        with TestClient(main.app) as client:
            sid = client.post("/api/sessions", json={"goal": "理解AQS"}).json()["id"]
            client.post(f"/api/sessions/{sid}/chat", json={"content": "并发基础是线程安全协调的基础"})
            s = db.get_session(sid)
            ai_msg = s["messages"][2]
            self.assertEqual(ai_msg["role"], "assistant")
            self.assertIsNone(ai_msg.get("diagnosis"))

    def test_chat_stream_messages_not_duplicated(self):
        """流式接口完整消费后，消息同样不重复。"""
        with TestClient(main.app) as client:
            sid = client.post("/api/sessions", json={"goal": "理解AQS"}).json()["id"]
            with client.stream(
                "POST", f"/api/sessions/{sid}/chat/stream",
                json={"content": "并发基础是线程安全协调的基础"},
            ) as resp:
                self.assertEqual(resp.status_code, 200)
                # 完整消费整个流
                for _ in resp.iter_lines():
                    pass
            s = db.get_session(sid)
            self.assertEqual(len(s["messages"]), 3)
            self.assertEqual(s["messages"][1]["content"], "并发基础是线程安全协调的基础")

    def test_user_message_survives_persist_failure(self):
        """模拟流式最后一步中断（等价于用户刷新时流未完成）：用户消息仍已落库。"""
        with TestClient(main.app, raise_server_exceptions=False) as client:
            sid = client.post("/api/sessions", json={"goal": "理解AQS"}).json()["id"]
            with mock.patch.object(main, "_persist", side_effect=RuntimeError("流中断")):
                resp = client.post(f"/api/sessions/{sid}/chat", json={"content": "并发基础"})
            self.assertEqual(resp.status_code, 500)
            s = db.get_session(sid)
            # 开场白 + 用户消息（AI 回复未追加）
            self.assertEqual(len(s["messages"]), 2)
            self.assertEqual(s["messages"][-1]["content"], "并发基础")


if __name__ == "__main__":
    unittest.main()
