"""「思考过程」trace 采集链路的回归测试。

覆盖三点：
1. llm._record_trace 正确追加记录（含 usage），trace=None 时安全跳过。
2. 各层（诊断/决策/回复）把 trace 参数透传给底层 chat_json / chat_text。
3. 端到端：/chat 返回 payload 携带 trace 字段（离线模式为空列表）。

强制 AI_ENABLED=False 走确定性路径，用 mock 验证透传，不依赖外部模型。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import config

config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "trace_test.db")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import cognitive  # noqa: E402
import tutor  # noqa: E402
import llm  # noqa: E402
from schemas import Concept, DiagnosticResult  # noqa: E402


def _concept() -> Concept:
    return Concept(id="c1", name="并发基础", summary="多线程协调的基础", why_matters="", prerequisites=[], common_misconceptions=[])


def _diag() -> DiagnosticResult:
    return DiagnosticResult(state="partial", confidence=0.5, concept_ids=["c1"], evidence="e", misconception="", missing=[])


class TestRecordTrace(unittest.TestCase):
    def test_record_trace_appends(self):
        trace: list = []
        usage = mock.Mock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.total_tokens = 15
        llm._record_trace(trace, "认知诊断", "model-x", "sys", "usr", "out", usage)
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["label"], "认知诊断")
        self.assertEqual(trace[0]["model"], "model-x")
        self.assertEqual(trace[0]["output"], "out")
        self.assertEqual(trace[0]["usage"]["total_tokens"], 15)

    def test_record_trace_skips_when_none(self):
        # trace=None 时不采集，且不抛异常
        llm._record_trace(None, "x", "m", "s", "u", "o", None)


class TestTracePropagation(unittest.TestCase):
    def test_diagnose_passes_trace(self):
        with mock.patch.object(cognitive, "chat_json", return_value=None) as m:
            trace: list = []
            cognitive.diagnose("目标", [_concept()], "一些内容", "c1", trace=trace)
            self.assertTrue(m.called)
            kwargs = m.call_args.kwargs
            self.assertIn("trace", kwargs)
            self.assertIs(kwargs["trace"], trace)
            self.assertEqual(kwargs["trace_label"], "认知诊断")

    def test_diagnose_and_decide_passes_trace(self):
        # 合并诊断+决策的 trace 透传：mock chat_json 验证 trace 参数被透传
        with mock.patch.object(cognitive, "chat_json", return_value=None) as m:
            trace: list = []
            cognitive.diagnose_and_decide("目标", [_concept()], "一些内容", "c1", trace=trace)
            self.assertTrue(m.called)
            kwargs = m.call_args.kwargs
            self.assertIn("trace", kwargs)
            self.assertIs(kwargs["trace"], trace)
            self.assertEqual(kwargs["trace_label"], "诊断+决策")

    def test_tutor_passes_trace(self):
        with mock.patch.object(tutor, "chat_text", return_value="回复") as m:
            trace: list = []
            tutor.generate_tutor_reply(_concept(), _diag(), "explain", "用户表达", trace=trace)
            self.assertTrue(m.called)
            kwargs = m.call_args.kwargs
            self.assertIn("trace", kwargs)
            self.assertEqual(kwargs["trace_label"], "回复生成")


class TestChatTraceField(unittest.TestCase):
    def test_chat_returns_trace_field(self):
        with TestClient(main.app) as client:
            r = client.post("/api/sessions", json={"goal": "理解AQS"})
            sid = r.json()["id"]
            r2 = client.post(
                f"/api/sessions/{sid}/chat",
                json={"content": "并发基础是线程安全协调的基础"},
            )
            self.assertEqual(r2.status_code, 200)
            body = r2.json()
            self.assertIn("trace", body)
            self.assertIsInstance(body["trace"], list)
            # 离线模式无 LLM 调用，trace 为空
            self.assertEqual(body["trace"], [])

    def test_chat_persists_trace_in_message(self):
        import db
        with TestClient(main.app) as client:
            r = client.post("/api/sessions", json={"goal": "理解AQS"})
            sid = r.json()["id"]
            client.post(f"/api/sessions/{sid}/chat", json={"content": "并发基础是线程安全协调的基础"})
            s = db.get_session(sid)
            # 最后一条 assistant 消息应携带 trace 字段（离线为空列表）
            last = s["messages"][-1]
            self.assertEqual(last["role"], "assistant")
            self.assertIn("trace", last)
            self.assertEqual(last["trace"], [])


if __name__ == "__main__":
    unittest.main()
