"""token 预算接入（Phase 2 · Task 2）的单元测试。

覆盖 _apply_budget：budget_label 覆盖 max_tokens、prompt 超限告警、
以及无 budget_label 时的行为等价（原样返回）。
"""
from __future__ import annotations

import unittest
from unittest import mock

import config

config.AI_ENABLED = False

import llm


class TestApplyBudget(unittest.TestCase):
    def test_no_label_returns_original(self):
        # 无 budget_label 时行为等价：原样返回 max_tokens
        self.assertEqual(llm._apply_budget(None, "sys", "usr", 4000), 4000)

    def test_unknown_label_returns_original(self):
        self.assertEqual(llm._apply_budget("no_such_layer", "sys", "usr", 123), 123)

    def test_label_overrides_max_tokens(self):
        # cognitive 层 completion=1500，应覆盖传入的 max_tokens
        self.assertEqual(llm._apply_budget("cognitive", "sys", "usr", 9999), 1500)

    def test_over_limit_warns(self):
        # 构造超限 prompt：cognitive prompt 上限 4000，用 5000+ 中文字符触发告警
        big = "好" * 5000
        with mock.patch.object(llm, "_warn_budget") as warn:
            llm._apply_budget("cognitive", big, "", 1500)
            warn.assert_called_once()
            # 告警参数：label + prompt_tokens + limit
            self.assertEqual(warn.call_args[0][0], "cognitive")
            self.assertGreater(warn.call_args[0][1], 4000)

    def test_under_limit_no_warn(self):
        small = "好" * 10
        with mock.patch.object(llm, "_warn_budget") as warn:
            llm._apply_budget("cognitive", small, "", 1500)
            warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
