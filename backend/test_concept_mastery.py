"""概念掌握档案 + 遗忘衰减（Phase 3 · Task 2）的单元测试。

覆盖 apply_forgetting（时间衰减 + 下限）、persist_concept_mastery、
prior_mastery（读历史 + 衰减）、以及 _build_cognitive 的历史先验。
"""
from __future__ import annotations

import os
import tempfile
import unittest

import config

config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "concept_mastery_test.db")

import db  # noqa: E402
import learner_profile  # noqa: E402
import main  # noqa: E402
from schemas import Concept  # noqa: E402


def _concept(cid: str, name: str) -> Concept:
    return Concept(
        id=cid, name=name, summary="s", why_matters="w",
        prerequisites=[], common_misconceptions=[],
    )


class TestApplyForgetting(unittest.TestCase):
    def test_no_decay_at_zero_days(self):
        self.assertAlmostEqual(learner_profile.apply_forgetting(0.8, 0.0), 0.8)

    def test_decays_over_time(self):
        self.assertAlmostEqual(learner_profile.apply_forgetting(0.8, 30.0), 0.8 / 2, places=3)

    def test_longer_time_more_decay(self):
        self.assertLess(
            learner_profile.apply_forgetting(0.8, 90.0),
            learner_profile.apply_forgetting(0.8, 30.0),
        )

    def test_has_floor(self):
        # 极长时间后衰减到下限，不衰减到 0
        self.assertGreaterEqual(
            learner_profile.apply_forgetting(0.8, 10000.0),
            learner_profile.FORGETTING_FLOOR - 1e-9,
        )


class TestPersistAndPrior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def test_persist_then_prior(self):
        cognitive = main._build_cognitive(
            "理解 HTTP",
            [_concept("http-basics", "HTTP 基础"), _concept("tls", "TLS")],
        )
        # 手动把掌握度设为已掌握再沉淀
        for m in cognitive["concepts"]:
            m["mastery"] = 0.9
        learner_profile.persist_concept_mastery(cognitive, "cm-u1")
        prior = learner_profile.prior_mastery("cm-u1")
        self.assertIn("http-basics", prior)
        self.assertIn("tls", prior)
        # 刚写入（0 天）几乎不衰减
        self.assertGreater(prior["http-basics"], 0.89)

    def test_prior_uses_forgetting(self):
        db.upsert_concept_mastery("cm-u2", "http-basics", 0.9)
        prior = learner_profile.prior_mastery("cm-u2")
        # 刚写入几乎不衰减；衰减后仍 >= floor
        self.assertGreater(prior["http-basics"], learner_profile.FORGETTING_FLOOR - 1e-9)

    def test_no_history_returns_empty(self):
        self.assertEqual(learner_profile.prior_mastery("no_such_user"), {})


class TestBuildCognitivePrior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def test_fallback_to_p0_when_no_history(self):
        cognitive = main._build_cognitive("理解 HTTP", [_concept("no-history-unique", "HTTP 基础")])
        # 无历史时回退 P_L0（intermediate 默认 0.35）
        self.assertEqual(cognitive["concepts"][0]["mastery"], config.P_L0)

    def test_uses_history_prior(self):
        db.upsert_concept_mastery(db.DEFAULT_USER_ID, "http-basics", 0.9)
        cognitive = main._build_cognitive("理解 HTTP", [_concept("http-basics", "HTTP 基础")])
        # 有历史时用历史掌握度（刚写入几乎不衰减，接近 0.9，但 > P_L0）
        self.assertGreater(cognitive["concepts"][0]["mastery"], config.P_L0)


if __name__ == "__main__":
    unittest.main()
