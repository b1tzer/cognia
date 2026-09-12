"""对话历史注入（概念级对话栈 Phase 1 · Task 1）的单元测试。

覆盖：
1. _diagnosis_history_block：仅在上轮为 probe 时注入追问上下文
2. _history_block：把对话轨迹渲染为 user prompt 片段
3. 诊断层 / 回复层端到端注入：history 非空时 prompt 包含上下文；history=None 时行为不变
"""
from __future__ import annotations

import unittest
from unittest import mock

import config

config.AI_ENABLED = False

import cognitive
import tutor
from schemas import Concept, DiagnosticResult


def _concept() -> Concept:
    return Concept(
        id="http-basics",
        name="HTTP 基础",
        summary="HTTP 是无状态请求-响应协议",
        why_matters="理解 Web 交互的前提",
        prerequisites=[],
        common_misconceptions=[],
    )


def _diag() -> DiagnosticResult:
    return DiagnosticResult(
        state="partial", confidence=0.6, concept_ids=["http-basics"],
        evidence="", misconception="", missing=[],
    )


def _history() -> list:
    return [
        {
            "user_text": "它是发请求收响应",
            "ai_reply": "那请求里包含哪些关键部分？",
            "action": "probe",
            "state": "partial",
        }
    ]


class TestDiagnosisHistoryBlock(unittest.TestCase):
    def test_none_history_empty(self):
        self.assertEqual(cognitive._diagnosis_history_block(None), "")

    def test_non_probe_history_empty(self):
        h = [{"user_text": "x", "ai_reply": "y", "action": "explain", "state": "partial"}]
        self.assertEqual(cognitive._diagnosis_history_block(h), "")

    def test_probe_history_injects_context(self):
        block = cognitive._diagnosis_history_block(_history())
        self.assertIn("上一轮追问", block)
        self.assertIn("那请求里包含哪些关键部分", block)
        self.assertIn("学习者上一轮回答", block)
        self.assertIn("它是发请求收响应", block)


class TestTutorHistoryBlock(unittest.TestCase):
    def test_none_history_empty(self):
        self.assertEqual(tutor._history_block(None), "")

    def test_history_renders_trajectory(self):
        block = tutor._history_block(_history())
        self.assertIn("导师", block)
        self.assertIn("那请求里包含哪些关键部分", block)
        self.assertIn("学习者", block)
        self.assertIn("它是发请求收响应", block)


class TestDiagnoseInjectsHistory(unittest.TestCase):
    def test_history_none_keeps_prompt_clean(self):
        captured = {}

        def fake_chat_json(system, user, **kwargs):
            captured["user"] = user
            return {"state": "partial", "confidence": 0.5, "concept_ids": [],
                    "evidence": "", "misconception": "", "missing": []}

        with mock.patch.object(cognitive, "chat_json", side_effect=fake_chat_json):
            cognitive._diagnose_with_llm(
                "理解 HTTP", [_concept()], "它是发请求收响应",
                focus_concept_id="http-basics", history=None,
            )
        self.assertNotIn("对话上下文", captured["user"])

    def test_history_injects_context(self):
        captured = {}

        def fake_chat_json(system, user, **kwargs):
            captured["user"] = user
            return {"state": "partial", "confidence": 0.5, "concept_ids": [],
                    "evidence": "", "misconception": "", "missing": []}

        with mock.patch.object(cognitive, "chat_json", side_effect=fake_chat_json):
            cognitive._diagnose_with_llm(
                "理解 HTTP", [_concept()], "就是那个",
                focus_concept_id="http-basics", history=_history(),
            )
        self.assertIn("上一轮追问", captured["user"])
        self.assertIn("学习者上一轮回答", captured["user"])


class TestTutorInjectsHistory(unittest.TestCase):
    def test_history_injected_into_user_prompt(self):
        captured = {}

        def fake_chat_text(system, user, **kwargs):
            captured["user"] = user
            return "好，我们继续"

        with mock.patch.object(tutor, "chat_text", side_effect=fake_chat_text):
            tutor._tutor_with_llm(
                _concept(), _diag(), "probe", user_text="就是那个",
                trace=None, history=_history(),
            )
        self.assertIn("近几轮对话轨迹", captured["user"])
        self.assertIn("它是发请求收响应", captured["user"])


if __name__ == "__main__":
    unittest.main()
