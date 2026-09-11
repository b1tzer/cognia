"""流式接口与流式生成器的回归测试。

覆盖：
1. tutor.stream_tutor_reply 在 AI 关闭时降级为一次性 yield 模板文本。
2. /chat/stream 接口返回 text/event-stream，含 token 与 done 事件，
   done 事件携带完整 payload（reply/action/diagnosis/status 等）。
3. 非流式 /chat 接口行为不回归（抽取 _process_turn 后保持一致）。

用标准库 unittest + mock，强制 AI_ENABLED=False 走确定性路径，不依赖外部模型。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import config

config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "stream_test.db")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import tutor  # noqa: E402
from schemas import Concept, DiagnosticResult  # noqa: E402


def _concept() -> Concept:
    return Concept(
        id="concurrent-basics",
        name="并发基础",
        summary="多线程安全协调的基础",
        why_matters="理解并发的根基",
        prerequisites=[],
        common_misconceptions=[],
    )


def _diag() -> DiagnosticResult:
    return DiagnosticResult(
        state="partial",
        confidence=0.5,
        concept_ids=["concurrent-basics"],
        evidence="并发基础",
        misconception="",
        missing=[],
    )


class TestStreamTutorReply(unittest.TestCase):
    def test_fallback_yields_template_once(self):
        # AI 关闭时 chat_text_stream 无输出 → 降级 yield 一次完整模板
        chunks = list(tutor.stream_tutor_reply(_concept(), _diag(), "explain"))
        self.assertEqual(len(chunks), 1)
        self.assertIn("并发基础", chunks[0])


class TestChatStreamEndpoint(unittest.TestCase):
    def test_stream_returns_sse(self):
        with TestClient(main.app) as client:
            r = client.post("/api/sessions", json={"goal": "理解AQS"})
            sid = r.json()["id"]

            with client.stream(
                "POST", f"/api/sessions/{sid}/chat/stream",
                json={"content": "并发基础是线程安全协调的基础"},
            ) as resp:
                self.assertEqual(resp.status_code, 200)
                self.assertIn("text/event-stream", resp.headers.get("content-type", ""))
                lines = [ln for ln in resp.iter_lines() if ln]

            # 至少包含一个 done 事件
            done_lines = [ln for ln in lines if "done" in ln]
            self.assertGreaterEqual(len(done_lines), 1)

            # 解析 done 事件的 payload，校验关键字段完整
            last = json.loads(done_lines[-1].removeprefix("data: "))
            self.assertEqual(last["type"], "done")
            data = last["data"]
            self.assertIn("reply", data)
            self.assertIn("action", data)
            self.assertIn("diagnosis", data)
            self.assertIn("status", data)
            self.assertEqual(data["session_id"], sid)

    def test_chat_non_stream_still_works(self):
        # 抽取 _process_turn 后，非流式 chat 接口行为不回归
        with TestClient(main.app) as client:
            r = client.post("/api/sessions", json={"goal": "理解AQS"})
            sid = r.json()["id"]
            r2 = client.post(
                f"/api/sessions/{sid}/chat",
                json={"content": "并发基础是线程安全协调的基础"},
            )
            self.assertEqual(r2.status_code, 200)
            body = r2.json()
            self.assertIn(body["action"], ("probe", "explain", "correct", "backtrack", "advance"))
            self.assertTrue(body["reply"])
            self.assertIn("state", body["diagnosis"])


if __name__ == "__main__":
    unittest.main()
