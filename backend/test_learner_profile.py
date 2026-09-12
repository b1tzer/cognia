"""学习者画像数据模型（Phase 3 · Task 1）的单元测试。

覆盖 learner_profile / concept_mastery / misconceptions 三张表的读写，
以及 init_db 建表幂等（老库可安全重复迁移）。
"""
from __future__ import annotations

import os
import tempfile
import unittest

import config

config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "learner_profile_test.db")

import db  # noqa: E402


class TestLearnerProfileSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def test_init_db_idempotent(self):
        # 重复 init_db 不报错（幂等迁移）
        db.init_db()
        db.init_db()

    def test_learner_profile_upsert_get(self):
        db.upsert_learner_profile("u1", prior_level="advanced", preferences={"style": "visual"})
        p = db.get_learner_profile("u1")
        self.assertEqual(p["prior_level"], "advanced")
        self.assertEqual(p["preferences"]["style"], "visual")
        # upsert 覆盖
        db.upsert_learner_profile("u1", prior_level="novice", preferences={"style": "verbal"})
        p = db.get_learner_profile("u1")
        self.assertEqual(p["prior_level"], "novice")
        self.assertEqual(p["preferences"]["style"], "verbal")

    def test_learner_profile_default_user(self):
        db.upsert_learner_profile(prior_level="intermediate")
        p = db.get_learner_profile()
        self.assertEqual(p["user_id"], db.DEFAULT_USER_ID)
        self.assertEqual(p["prior_level"], "intermediate")

    def test_concept_mastery_upsert_get(self):
        db.upsert_concept_mastery("u1", "http-basics", 0.8, "证据A")
        db.upsert_concept_mastery("u1", "tls", 0.5)
        m = db.get_concept_mastery("u1")
        self.assertEqual(m["http-basics"]["mastery"], 0.8)
        self.assertEqual(m["http-basics"]["last_evidence"], "证据A")
        self.assertEqual(m["tls"]["mastery"], 0.5)
        # upsert 覆盖
        db.upsert_concept_mastery("u1", "http-basics", 0.9)
        self.assertEqual(db.get_concept_mastery("u1")["http-basics"]["mastery"], 0.9)

    def test_misconception_add_get_filter(self):
        db.add_misconception("u1", "http-basics", "以为 HTTP 是加密的", 0.7)
        db.add_misconception("u1", "tls", "以为 HTTPS 隐藏所有信息", 0.6)
        all_m = db.get_misconceptions("u1")
        self.assertEqual(len(all_m), 2)
        only_http = db.get_misconceptions("u1", concept_id="http-basics")
        self.assertEqual(len(only_http), 1)
        self.assertEqual(only_http[0]["misconception"], "以为 HTTP 是加密的")

    def test_user_isolation(self):
        db.upsert_concept_mastery("u1", "http-basics", 0.8)
        # 不同 user_id 互不可见
        self.assertEqual(db.get_concept_mastery("u2"), {})


if __name__ == "__main__":
    unittest.main()
