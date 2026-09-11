"""prompt 泄漏修复的回归测试。

覆盖两处根因：
1. cognitive._heuristic_diagnose 的 evidence 字段只引用学习者原话，
   绝不写入「启发式判定 / 证据不足」等引擎自述文案。
2. tutor._tutor_with_llm 的「学习者表达」来自真实 user_text，
   不再用 evidence 反推（即使 evidence 被污染也不串台到回复层）。

用标准库 unittest + mock，不引入额外依赖。强制 AI_ENABLED=False 走确定性路径。
"""
from __future__ import annotations

import unittest
from unittest import mock

import config

# 强制离线路径（保证诊断走启发式，测试确定性）
config.AI_ENABLED = False

import cognitive
import tutor
from schemas import Concept, DiagnosticResult


def _thread_concepts() -> list[Concept]:
    return [
        Concept(
            id="thread",
            name="线程",
            summary="CPU 调度的最小执行单元",
            why_matters="理解并发的基础",
            prerequisites=[],
            common_misconceptions=[],
        )
    ]


class TestHeuristicEvidenceNoLeak(unittest.TestCase):
    def test_partial_hit_concept_uses_snippet(self):
        text = "线程是 CPU 调度的最小单位，比进程更轻量，可以共享进程资源"
        r = cognitive._heuristic_diagnose(_thread_concepts(), text, "thread")
        self.assertEqual(r.state, "partial")
        # evidence 是原话截断，而非引擎自述
        self.assertEqual(r.evidence, text[:60])
        self.assertNotIn("启发式", r.evidence)
        self.assertNotIn("证据不足", r.evidence)
        self.assertNotIn("离线模式", r.evidence)

    def test_default_partial_uses_snippet(self):
        # 不含概念词、长度足够、无否定信号 → 走默认半理解分支
        text = "这是一段不含概念词但长度足够用于测试的回答内容"
        r = cognitive._heuristic_diagnose(_thread_concepts(), text, "thread")
        self.assertEqual(r.state, "partial")
        self.assertEqual(r.evidence, text[:60])
        self.assertNotIn("启发式判定", r.evidence)

    def test_insufficient_uses_snippet(self):
        # 信息不足分支：evidence 也不再是「表达空泛或明确表示不清楚」
        r = cognitive._heuristic_diagnose(_thread_concepts(), "不知道", "thread")
        self.assertEqual(r.state, "insufficient")
        self.assertEqual(r.evidence, "不知道")
        self.assertNotIn("表达空泛", r.evidence)

    def test_no_internal_phrases_anywhere(self):
        # 遍历多组输入，确保 evidence 永不含内部实现文案
        cases = [
            "线程是 CPU 调度的最小单位，比进程更轻量，可以共享进程资源",
            "这是一段不含概念词但长度足够用于测试的回答内容",
            "不知道",
            "我以为线程就是进程",
        ]
        internal_phrases = ("启发式", "证据不足", "离线模式", "表达空泛", "引擎")
        for c in cases:
            r = cognitive._heuristic_diagnose(_thread_concepts(), c, "thread")
            for phrase in internal_phrases:
                self.assertNotIn(phrase, r.evidence, f"evidence 泄漏内部文案: {r.evidence}")


class TestTutorUsesUserText(unittest.TestCase):
    def test_llm_uses_user_text_not_evidence(self):
        captured = {}

        def fake_chat_text(system, user, temperature=0.7, max_tokens=2000):
            captured["user"] = user
            return "一条回复"

        concept = _thread_concepts()[0]
        # 即使 evidence 被污染为引擎自述，tutor 也应优先用真实 user_text
        diag = DiagnosticResult(
            state="partial",
            confidence=0.5,
            concept_ids=["thread"],
            evidence="启发式判定，证据不足",
            misconception="",
            missing=[],
        )
        with mock.patch.object(tutor, "chat_text", side_effect=fake_chat_text):
            reply = tutor._tutor_with_llm(concept, diag, "explain", user_text="线程是 CPU 调度的最小单位")

        self.assertEqual(reply, "一条回复")
        self.assertIn("线程是 CPU 调度的最小单位", captured["user"])
        self.assertNotIn("启发式判定", captured["user"])

    def test_generate_tutor_reply_passes_user_text(self):
        # 完整链路：generate_tutor_reply 把 user_text 透传给 _tutor_with_llm
        concept = _thread_concepts()[0]
        diag = DiagnosticResult(
            state="partial",
            confidence=0.5,
            concept_ids=["thread"],
            evidence="被污染的 evidence",
            misconception="",
            missing=[],
        )
        with mock.patch.object(tutor, "chat_text", return_value="真实回复"):
            reply = tutor.generate_tutor_reply(concept, diag, "probe", user_text="我的真实回答")

        self.assertEqual(reply, "真实回复")


if __name__ == "__main__":
    unittest.main()
