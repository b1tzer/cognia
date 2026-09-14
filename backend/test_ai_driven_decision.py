"""需求 #71 需求1：教学判断交还 AI 的测试。

验证核心语义变化：推进由 LLM 输出的 action 决定，而非由 diagnosis.state 反推。
- action=advance 时推进（即使 state 不是 understood）
- action!=advance 时不推进（即使 state 是 understood）
"""
from __future__ import annotations

import unittest
from unittest import mock

import config
config.AI_ENABLED = False

import cognitive as cog
import learner_profile
import main
from schemas import ActionDecision, ActionReason, Concept, DiagnosticResult


KNOWLEDGE = {
    "goal": "测试目标",
    "root_concepts": ["c"],
    "concepts": [
        Concept(id="a", name="概念A", summary="", why_matters="",
                prerequisites=[], common_misconceptions=[]).model_dump(),
        Concept(id="b", name="概念B", summary="", why_matters="",
                prerequisites=["a"], common_misconceptions=[]).model_dump(),
        Concept(id="c", name="概念C", summary="", why_matters="",
                prerequisites=["b"], common_misconceptions=[]).model_dump(),
    ],
}


def _make_session() -> dict:
    concepts = [Concept(**c) for c in KNOWLEDGE["concepts"]]
    cognitive = main._build_cognitive("测试目标", concepts)
    return {"knowledge": KNOWLEDGE, "cognitive": cognitive, "status": "active"}


class TestAdvanceByAction(unittest.TestCase):
    def setUp(self):
        p0 = mock.patch.object(learner_profile, "prior_mastery", return_value={})
        p1 = mock.patch.object(learner_profile, "get_misconceptions_for", return_value={})
        p2 = mock.patch.object(learner_profile, "record_misconception", return_value=None)
        p0.start()
        p1.start()
        p2.start()
        self.addCleanup(p0.stop)
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def test_advance_action_forces_advance_even_if_state_partial(self):
        """LLM 输出 action=advance 时，即使 state=partial 也应推进（推进看 action 不看 state）。"""
        s = _make_session()
        diag = DiagnosticResult(state="partial", confidence=0.5, concept_ids=["a"],
                                evidence="e", misconception="", missing=[], quality="")
        ad = ActionDecision(chosen_action="advance", reasons=ActionReason())
        with mock.patch.object(cog, "diagnose_and_decide", return_value=(diag, ad)):
            ctx = main._process_turn(s, "我觉得这个概念我理解了")
        self.assertTrue(ctx["should_advance"])
        self.assertEqual(ctx["action"], "advance")

    def test_understood_state_does_not_advance_without_advance_action(self):
        """state=understood 但 LLM 输出 action=probe 时，不应推进（推进不再由 state 反推）。"""
        s = _make_session()
        diag = DiagnosticResult(state="understood", confidence=0.9, concept_ids=["a"],
                                evidence="e", misconception="", missing=[], quality="deep")
        ad = ActionDecision(chosen_action="probe", reasons=ActionReason())
        with mock.patch.object(cog, "diagnose_and_decide", return_value=(diag, ad)):
            ctx = main._process_turn(s, "我明白了")
        self.assertFalse(ctx["should_advance"])
        self.assertEqual(ctx["action"], "probe")


if __name__ == "__main__":
    unittest.main()
