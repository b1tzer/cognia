"""推进链路端到端测试：验证「明确推进 / 步数上限 / LLM 判定 understood」三种
推进来源都能让 mastered 置位、焦点前进。

这是防「诊断准但推进卡死」死循环回归的关键测试。此前 advance_intent 只写
mastery、is_mastered 却要求三层数值 AND（mastery+success_count+quality），
生产方与消费方字段错位，导致焦点永远卡在第一个概念（Java LockSupport 会话
17 轮原地打转）。
"""
from __future__ import annotations

import unittest
from unittest import mock

import config
config.AI_ENABLED = False

import cognitive as cog
import decision
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


def _mastered(s: dict, cid: str) -> bool:
    for m in s["cognitive"]["concepts"]:
        if m["concept_id"] == cid:
            return bool(m.get("mastered"))
    return False


class TestAdvanceE2E(unittest.TestCase):
    def setUp(self):
        # 隔离 db 依赖（_build_cognitive 会查 prior_mastery，learner_profile 会查/写长期记忆表）
        p0 = mock.patch.object(learner_profile, "prior_mastery", return_value={})
        p1 = mock.patch.object(learner_profile, "get_misconceptions_for", return_value={})
        p2 = mock.patch.object(learner_profile, "record_misconception", return_value=None)
        p0.start()
        p1.start()
        p2.start()
        self.addCleanup(p0.stop)
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)

    def test_explicit_advance_marks_mastered_and_advances(self):
        s = _make_session()
        ctx = main._process_turn(s, "我懂了，继续下一个")
        self.assertTrue(ctx["should_advance"])
        self.assertEqual(ctx["focus_id"], "a")
        self.assertTrue(_mastered(s, "a"), "推进后当前焦点必须 mastered")
        next_focus = decision.next_focus_concept(ctx["knowledge"], ctx["cognitive"])
        self.assertEqual(next_focus["id"], "b")

    def test_step_limit_forces_advance(self):
        s = _make_session()
        for m in s["cognitive"]["concepts"]:
            if m["concept_id"] == "a":
                m["evidence_count"] = config.MAX_STEPS_PER_CONCEPT
        ctx = main._process_turn(s, "我不知道")
        self.assertTrue(ctx["should_advance"])
        self.assertTrue(_mastered(s, "a"))

    def test_understood_advances(self):
        s = _make_session()
        diag = DiagnosticResult(
            state="understood", confidence=0.9, concept_ids=["a"],
            evidence="说得对", misconception="", missing=[], quality="deep",
        )
        ad = decision.ActionDecision(chosen_action="advance", reasons=decision.ActionReason())
        with mock.patch.object(cog, "diagnose_and_decide", return_value=(diag, ad)):
            ctx = main._process_turn(s, "线程有就绪、可运行、阻塞、等待等状态")
        self.assertTrue(ctx["should_advance"])
        self.assertTrue(_mastered(s, "a"))

    def test_partial_does_not_advance(self):
        s = _make_session()
        diag = DiagnosticResult(
            state="partial", confidence=0.7, concept_ids=["a"],
            evidence="方向对但有遗漏", misconception="", missing=[], quality="",
        )
        ad = decision.ActionDecision(chosen_action="probe", reasons=decision.ActionReason())
        with mock.patch.object(cog, "diagnose_and_decide", return_value=(diag, ad)):
            ctx = main._process_turn(s, "线程有就绪、可运行、阻塞等状态")
        self.assertFalse(ctx["should_advance"])
        self.assertFalse(_mastered(s, "a"))
        # 动作必须直接来自合并调用产出的 action_decision，而非二次决策
        self.assertEqual(ctx["action"], "probe")


if __name__ == "__main__":
    unittest.main()
