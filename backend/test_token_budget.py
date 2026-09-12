"""显式 token 预算工具（Phase 2 · Task 1）的单元测试。

覆盖 estimate_tokens（字符估算）、TokenBudget（reserve→add→drop+告警）、
以及 _usage 的分层（by_layer）统计。
"""
from __future__ import annotations

import unittest

import config

config.AI_ENABLED = False

import llm


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(llm.estimate_tokens(""), 0)
        self.assertEqual(llm.estimate_tokens(None), 0)

    def test_pure_chinese(self):
        # 中文 1 字 ≈ 1 token
        self.assertEqual(llm.estimate_tokens("你好世界"), 4)

    def test_pure_english(self):
        # 英文 4 字符 ≈ 1 token，向上取整
        self.assertEqual(llm.estimate_tokens("abcd"), 1)
        self.assertEqual(llm.estimate_tokens("abcde"), 2)

    def test_mixed(self):
        # 混合文本 = 中文字数 + 英文字符/4 向上取整
        # "AB你好"：中文 2 字 + 英文 2 字符 → 2 + ceil(2/4)=2+1=3
        self.assertEqual(llm.estimate_tokens("AB你好"), 3)


class TestTokenBudget(unittest.TestCase):
    def test_add_within_budget(self):
        b = llm.TokenBudget(limit=100, reserve_output=20)
        self.assertTrue(b.add("system", "你" * 10))  # 10 tokens（中文 1 字 1 token）
        self.assertTrue(b.add("user", "好" * 5))      # 5 tokens
        self.assertEqual(b.remaining, 100 - 20 - 15)

    def test_drop_when_over_budget(self):
        b = llm.TokenBudget(limit=10, reserve_output=0)
        self.assertFalse(b.add("big", "x" * 100))
        self.assertIn("big", b.dropped)

    def test_reserve_output_reduces_remaining(self):
        b = llm.TokenBudget(limit=100, reserve_output=40)
        self.assertEqual(b.remaining, 60)

    def test_usage_ratio_and_warn(self):
        b = llm.TokenBudget(limit=100, reserve_output=0)
        b.add("a", "好" * 85)  # 85 tokens
        self.assertGreaterEqual(b.usage_ratio, 0.8)
        self.assertTrue(b.warn_ratio_exceeded(0.8))


class TestUsageByLayer(unittest.TestCase):
    def test_record_usage_by_layer(self):
        llm.reset_usage()
        resp = type("R", (), {"usage": type("U", (), {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        })()})()
        llm._record_usage(resp, "认知诊断")
        llm._record_usage(resp, "认知诊断")
        llm._record_usage(resp, "回复生成")
        usage = llm.get_usage()
        self.assertEqual(usage["calls"], 3)
        self.assertEqual(usage["prompt_tokens"], 30)
        self.assertIn("认知诊断", usage["by_layer"])
        self.assertIn("回复生成", usage["by_layer"])
        self.assertEqual(usage["by_layer"]["认知诊断"]["calls"], 2)
        self.assertEqual(usage["by_layer"]["回复生成"]["calls"], 1)

    def test_reset_clears_by_layer(self):
        llm.reset_usage()
        resp = type("R", (), {"usage": type("U", (), {
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
        })()})()
        llm._record_usage(resp, "认知诊断")
        llm.reset_usage()
        self.assertEqual(llm.get_usage()["by_layer"], {})


if __name__ == "__main__":
    unittest.main()
