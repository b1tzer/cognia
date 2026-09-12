"""误解档案（Phase 3 · Task 3）的单元测试。

覆盖 record_misconception 去重、get_misconceptions_for 读取、
_misconception_block 注入格式，以及诊断端到端（misconceived 时写入）。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import config

config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "misconception_bank_test.db")

import cognitive  # noqa: E402
import db  # noqa: E402
import learner_profile  # noqa: E402


class TestMisconceptionBank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def test_record_deduplicates(self):
        learner_profile.record_misconception("http-basics", "以为 HTTP 是加密的", user_id="mc-u1")
        learner_profile.record_misconception("http-basics", "以为 HTTP 是加密的", user_id="mc-u1")
        rows = db.get_misconceptions("mc-u1", concept_id="http-basics")
        self.assertEqual(len(rows), 1)

    def test_get_misconceptions_for(self):
        learner_profile.record_misconception("http-basics", "误解A", user_id="mc-u2")
        learner_profile.record_misconception("http-basics", "误解B", user_id="mc-u2")
        learner_profile.record_misconception("tls", "误解C", user_id="mc-u2")
        out = learner_profile.get_misconceptions_for(["http-basics", "tls"], "mc-u2")
        self.assertEqual(out["http-basics"], ["误解B", "误解A"])  # 倒序
        self.assertEqual(out["tls"], ["误解C"])

    def test_misconception_block_empty(self):
        self.assertEqual(cognitive._misconception_block(None), "")
        self.assertEqual(cognitive._misconception_block({}), "")

    def test_misconception_block_format(self):
        block = cognitive._misconception_block({"http-basics": ["误解A", "误解B"]})
        self.assertIn("历史误解参考", block)
        self.assertIn("误解A；误解B", block)

    def test_diagnose_receives_misconceptions(self):
        captured = {}

        def fake_chat_json(system, user, **kwargs):
            captured["user"] = user
            return {"state": "partial", "confidence": 0.5, "concept_ids": [],
                    "evidence": "", "misconception": "", "missing": []}

        with mock.patch.object(cognitive, "chat_json", side_effect=fake_chat_json):
            from schemas import Concept
            c = Concept(id="http-basics", name="HTTP 基础", summary="s",
                        why_matters="w", prerequisites=[], common_misconceptions=[])
            cognitive._diagnose_with_llm(
                "理解 HTTP", [c], "HTTP 是加密的",
                focus_concept_id="http-basics", misconceptions={"http-basics": ["误解A"]},
            )
        self.assertIn("历史误解参考", captured["user"])
        self.assertIn("误解A", captured["user"])


if __name__ == "__main__":
    unittest.main()
