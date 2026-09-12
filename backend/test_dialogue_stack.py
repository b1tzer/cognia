"""概念级对话栈（Phase 1 · Task 2）的单元测试。

覆盖 _push_dialogue / _focus_history / _build_cognitive 的栈维护逻辑：
追加、容量截断、字段截断、空焦点跳过、按概念隔离。
"""
from __future__ import annotations

import unittest

import config

config.AI_ENABLED = False

import main
from schemas import Concept


def _concept(cid: str = "http-basics", name: str = "HTTP 基础") -> Concept:
    return Concept(
        id=cid, name=name, summary="HTTP 是无状态协议",
        why_matters="理解 Web", prerequisites=[], common_misconceptions=[],
    )


def _cognitive() -> dict:
    return main._build_cognitive("理解 HTTP", [_concept()])


class TestBuildCognitiveDialogue(unittest.TestCase):
    def test_each_concept_has_dialogue(self):
        c = _cognitive()
        for m in c["concepts"]:
            self.assertIn("dialogue", m)
            self.assertEqual(m["dialogue"], [])


class TestPushDialogue(unittest.TestCase):
    def test_append_one_round(self):
        c = _cognitive()
        main._push_dialogue(c, "http-basics", "user", "ai", "probe", "partial")
        self.assertEqual(len(c["concepts"][0]["dialogue"]), 1)
        item = c["concepts"][0]["dialogue"][0]
        self.assertEqual(item["user_text"], "user")
        self.assertEqual(item["ai_reply"], "ai")
        self.assertEqual(item["action"], "probe")
        self.assertEqual(item["state"], "partial")

    def test_capacity_truncates_oldest(self):
        c = _cognitive()
        for i in range(5):
            main._push_dialogue(c, "http-basics", f"u{i}", f"a{i}", "probe", "partial")
        stack = c["concepts"][0]["dialogue"]
        self.assertEqual(len(stack), 3)
        # 只保留最近 3 轮：u2/u3/u4
        self.assertEqual([x["user_text"] for x in stack], ["u2", "u3", "u4"])

    def test_field_truncation(self):
        c = _cognitive()
        main._push_dialogue(c, "http-basics", "u" * 500, "a" * 500, "probe", "partial")
        item = c["concepts"][0]["dialogue"][0]
        self.assertEqual(len(item["user_text"]), main._DIALOGUE_USER_MAX)
        self.assertEqual(len(item["ai_reply"]), main._DIALOGUE_AI_MAX)

    def test_none_focus_skips(self):
        c = _cognitive()
        main._push_dialogue(c, None, "u", "a", "probe", "partial")
        self.assertEqual(c["concepts"][0]["dialogue"], [])

    def test_unknown_focus_skips(self):
        c = _cognitive()
        main._push_dialogue(c, "no-such", "u", "a", "probe", "partial")
        self.assertEqual(c["concepts"][0]["dialogue"], [])


class TestFocusHistory(unittest.TestCase):
    def test_returns_stack_for_focus(self):
        c = _cognitive()
        main._push_dialogue(c, "http-basics", "u", "a", "probe", "partial")
        h = main._focus_history(c, "http-basics")
        self.assertEqual(len(h), 1)

    def test_empty_for_none(self):
        c = _cognitive()
        self.assertEqual(main._focus_history(c, None), [])

    def test_empty_for_unknown(self):
        c = _cognitive()
        self.assertEqual(main._focus_history(c, "no-such"), [])


if __name__ == "__main__":
    unittest.main()
