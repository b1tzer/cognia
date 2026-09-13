"""诊断+动作合并（Map+Guide 单循环）单元测试。

验证 diagnose_and_decide 一次 LLM 调用同时产出诊断与动作，且动作与 state
在确定性候选集内保持一致；非法动作回退规则；LLM 不可用降级到启发式+规则。
"""
from __future__ import annotations

import unittest
from unittest import mock

import cognitive as cog
from schemas import Concept


def _concept() -> Concept:
    return Concept(
        id="a", name="概念A", summary="概念A摘要", why_matters="重要",
        prerequisites=[], common_misconceptions=[],
    )


class TestDiagnoseAndDecide(unittest.TestCase):
    def test_llm_called_once_and_returns_both(self):
        captured = {}
        data = {
            "state": "partial", "confidence": 0.7, "concept_ids": ["a"],
            "evidence": "方向对", "misconception": "", "missing": [],
            "quality": "", "action": "probe", "action_reason": "追问补全",
        }

        def fake(system, user, **kwargs):
            captured["system"] = system
            captured["user"] = user
            return data

        with mock.patch.object(cog, "chat_json", side_effect=fake) as m:
            diag, ad = cog.diagnose_and_decide(
                "目标", [_concept()], "我的理解", focus_concept_id="a",
                evidence_count=0, consecutive_failures=0,
            )

        # 核心诉求：一次 LLM 调用同时产出诊断与动作（不再是 diagnose + decide 两次）
        self.assertEqual(m.call_count, 1)
        self.assertEqual(diag.state, "partial")
        self.assertEqual(ad.chosen_action, "probe")
        # 合并 prompt 同时含诊断规则与动作规则
        self.assertIn("认知状态四分类", captured["system"])
        self.assertIn("教学动作选择", captured["system"])

    def test_action_must_be_legal_for_state(self):
        # state=misconceived 的合法动作是 correct/backtrack，不含 advance
        data = {
            "state": "misconceived", "confidence": 0.8, "concept_ids": ["a"],
            "evidence": "混淆了", "misconception": "搞混了", "missing": [],
            "quality": "", "action": "advance", "action_reason": "乱选",
        }
        with mock.patch.object(cog, "chat_json", return_value=data):
            diag, ad = cog.diagnose_and_decide(
                "目标", [_concept()], "我的错误理解", focus_concept_id="a",
                evidence_count=0, consecutive_failures=0,
            )
        self.assertEqual(diag.state, "misconceived")
        # advance 非法（不在 misconceived 候选集内），回退到规则首个 = correct
        self.assertEqual(ad.chosen_action, "correct")

    def test_llm_unavailable_falls_back(self):
        with mock.patch.object(cog, "chat_json", return_value=None):
            diag, ad = cog.diagnose_and_decide(
                "目标", [_concept()], "我不知道", focus_concept_id="a",
                evidence_count=0, consecutive_failures=0,
            )
        # 否定信号 → 启发式 insufficient；insufficient 候选集首个 = probe
        self.assertEqual(diag.state, "insufficient")
        self.assertEqual(ad.chosen_action, "probe")


if __name__ == "__main__":
    unittest.main()
