"""诊断层启发式误判 + 推进意图识别的回归测试。

覆盖三条根因链：
1. 「不会」等正常否定表述不再被启发式误判 insufficient（活 bug 3 核心）。
2. 复合概念名（如「并发基础」）能被核心词（「并发」）命中（活 bug 3 次要）。
3. LLM 诊断接收并标注焦点概念，避免概念错位（活 bug 2）。
4. 「继续/下一个」推进意图被正确识别（活 bug 1 的前置规则）。

用标准库 unittest + mock，不引入额外依赖。强制 AI_ENABLED=False 走确定性路径。
"""
from __future__ import annotations

import unittest
from unittest import mock

import config

# 强制离线路径，保证启发式诊断确定性
config.AI_ENABLED = False

import cognitive
import decision
from schemas import Concept


def _concept(name: str = "并发基础", cid: str = "concurrent-basics") -> Concept:
    return Concept(
        id=cid,
        name=name,
        summary="多线程安全协调的基础",
        why_matters="理解并发的根基",
        prerequisites=[],
        common_misconceptions=[],
    )


class TestConceptHits(unittest.TestCase):
    def test_full_name_hit(self):
        self.assertEqual(cognitive._concept_hits([_concept()], "并发基础是并发编程的根基"), 1)

    def test_core_word_hit(self):
        # 只含核心词「并发」，不含完整「并发基础」，也应命中
        self.assertEqual(cognitive._concept_hits([_concept()], "并发是多个线程同时执行"), 1)

    def test_no_hit(self):
        self.assertEqual(cognitive._concept_hits([_concept()], "这是一个完全无关的话题"), 0)


class TestHeuristicNoMisjudge(unittest.TestCase):
    def test_normal_negation_not_insufficient(self):
        # 修复前「不会」在 _UNKNOWN_HINTS 里，导致此句被判 insufficient
        text = "线程修改数据后，其他线程不会立马知道，需要及时通知"
        r = cognitive._heuristic_diagnose([_concept()], text, "concurrent-basics")
        self.assertNotEqual(r.state, "insufficient")

    def test_short_text_still_insufficient(self):
        # 过短表达仍应判 insufficient（未误伤该语义）
        r = cognitive._heuristic_diagnose([_concept()], "不知道", "concurrent-basics")
        self.assertEqual(r.state, "insufficient")


class TestDetectAdvanceIntent(unittest.TestCase):
    def test_advance_signals(self):
        for t in ("继续下一个", "我懂了，继续下一个", "下一个", "接着讲", "往下", "进入下一个"):
            self.assertTrue(decision.detect_advance_intent(t), f"应识别推进意图: {t}")

    def test_long_text_not_advance(self):
        # 长文本是真正的理解陈述，不应被当作推进指令
        text = "线程是 CPU 调度的最小执行单元，比进程更轻量，共享进程资源"
        self.assertFalse(decision.detect_advance_intent(text))

    def test_ambiguous_continue_not_advance(self):
        # 「继续解释」这类不含明确推进词的表达，不触发 advance
        self.assertFalse(decision.detect_advance_intent("继续解释一下这个概念"))


class TestDiagnoseLlmmFocus(unittest.TestCase):
    def test_llm_receives_focus_concept(self):
        captured = {}

        def fake_chat_json(system, user, temperature=0.3, max_tokens=4000, **kwargs):
            captured["user"] = user
            return {
                "state": "partial", "confidence": 0.5,
                "concept_ids": ["concurrent-basics"], "evidence": "e",
                "misconception": "", "missing": [],
            }

        with mock.patch.object(cognitive, "chat_json", side_effect=fake_chat_json):
            cognitive._diagnose_with_llm(
                "理解AQS", [_concept()], "并发是多个线程同时执行", focus_concept_id="concurrent-basics"
            )
        self.assertIn("当前诊断焦点：并发基础", captured["user"])


class TestDiagnoseQuality(unittest.TestCase):
    def test_understood_with_deep_quality(self):
        def fake_chat_json(system, user, temperature=0.3, max_tokens=4000, **kwargs):
            return {
                "state": "understood", "confidence": 0.9,
                "concept_ids": ["concurrent-basics"], "evidence": "e",
                "misconception": "", "missing": [], "quality": "deep",
            }
        with mock.patch.object(cognitive, "chat_json", side_effect=fake_chat_json):
            r = cognitive._diagnose_with_llm(
                "理解AQS", [_concept()], "并发是多个线程同时执行", focus_concept_id="concurrent-basics"
            )
        self.assertEqual(r.quality, "deep")

    def test_partial_forces_empty_quality(self):
        # LLM 误返回 quality=deep，但 state=partial → 强制置空
        def fake_chat_json(system, user, temperature=0.3, max_tokens=4000, **kwargs):
            return {
                "state": "partial", "confidence": 0.5,
                "concept_ids": ["concurrent-basics"], "evidence": "e",
                "misconception": "", "missing": [], "quality": "deep",
            }
        with mock.patch.object(cognitive, "chat_json", side_effect=fake_chat_json):
            r = cognitive._diagnose_with_llm(
                "理解AQS", [_concept()], "并发是多个线程同时执行", focus_concept_id="concurrent-basics"
            )
        self.assertEqual(r.quality, "")


if __name__ == "__main__":
    unittest.main()
