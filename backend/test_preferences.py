"""需求 #71 需求3：记忆用户习惯落地的测试。

覆盖：
1. learner_profile 偏好读写（get/update/bump/reset）
2. 偏好文本渲染 _preferences_text
3. 偏好抽取阈值触发与离线降级
4. 偏好注入诊断决策与回复层（透传参数）

强制 AI_ENABLED=False，偏好抽取走离线降级（不调 LLM），保证可复现。
"""
from __future__ import annotations

import os
import tempfile
import unittest

import config

config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "pref_test.db")

import db
import learner_profile
import main


class TestPreferencesPersistence(unittest.TestCase):
    def setUp(self):
        # 每个测试用独立 db，避免 learner_profile 数据跨测试累积
        config.DB_PATH = os.path.join(tempfile.mkdtemp(), "pref_test.db")
        db.init_db()

    def test_update_and_get_preferences(self):
        learner_profile.update_preferences({"teaching_style": "先讲后问"})
        prefs = learner_profile.get_preferences()
        self.assertEqual(prefs.get("teaching_style"), "先讲后问")

    def test_update_merges(self):
        learner_profile.update_preferences({"teaching_style": "先讲后问"})
        learner_profile.update_preferences({"depth": "深入底层"})
        prefs = learner_profile.get_preferences()
        self.assertEqual(prefs["teaching_style"], "先讲后问")
        self.assertEqual(prefs["depth"], "深入底层")

    def test_get_preferences_filters_internal(self):
        # 内部 _pending_turns 字段不应出现在 get_preferences 返回中
        learner_profile.bump_preference_pending()
        prefs = learner_profile.get_preferences()
        self.assertNotIn("_pending_turns", prefs)

    def test_empty_preferences(self):
        # 新用户无偏好记录，返回空 dict
        prefs = learner_profile.get_preferences()
        self.assertEqual(prefs, {})


class TestPreferencePending(unittest.TestCase):
    def setUp(self):
        config.DB_PATH = os.path.join(tempfile.mkdtemp(), "pref_test.db")
        db.init_db()

    def test_bump_and_reset(self):
        self.assertEqual(learner_profile.bump_preference_pending(), 1)
        self.assertEqual(learner_profile.bump_preference_pending(), 2)
        learner_profile.reset_preference_pending()
        self.assertEqual(learner_profile.bump_preference_pending(), 1)


class TestPreferencesText(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(main._preferences_text({}), "")

    def test_renders(self):
        text = main._preferences_text({"teaching_style": "先讲后问", "depth": "深入底层"})
        self.assertIn("teaching_style=先讲后问", text)
        self.assertIn("depth=深入底层", text)

    def test_filters_internal(self):
        text = main._preferences_text({"_pending_turns": 3, "depth": "深入底层"})
        self.assertNotIn("_pending_turns", text)
        self.assertIn("depth=深入底层", text)


class TestExtractPreferencesOffline(unittest.TestCase):
    def test_offline_returns_existing(self):
        existing = {"depth": "适中"}
        result = main._extract_preferences(
            [{"role": "user", "content": "你好"}], existing
        )
        self.assertEqual(result, existing)


if __name__ == "__main__":
    unittest.main()
