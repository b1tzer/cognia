"""零基础用户求助直达讲解 + 领域全景 的测试（需求 #69）。

覆盖：
1. detect_help_intent 识别求助信号（强/弱信号，不误伤推进/正常陈述）
2. _process_turn 用户说「我不懂」→ action=explain，不推进，诊断诚实不篡改
3. tutor._dag_overview 渲染领域全景
4. tutor._tutor_system explain 时注入全景（probe 时不注入）
5. tutor.build_intro 对 novice 开场先给全景，非 novice 保持先问后教

强制 AI_ENABLED=False 走确定性路径，保证可复现、不依赖外部模型。
"""
from __future__ import annotations

import unittest

import config

config.AI_ENABLED = False

import decision
import domain_model
import main
import tutor
from schemas import Concept, DiagnosticResult


def _make_session(goal: str = "理解 HTTP") -> dict:
    km = domain_model.build_knowledge_model(goal)
    cognitive = main._build_cognitive(goal, km.concepts)
    return {
        "knowledge": km.model_dump(),
        "cognitive": cognitive,
        "status": "active",
    }


class TestDetectHelpIntent(unittest.TestCase):
    def test_button_text_triggers(self):
        # 前端「我不懂，解释一下」按钮发送的真实文本
        self.assertTrue(decision.detect_help_intent("我还不懂，能解释一下吗？"))

    def test_short_unknown_triggers(self):
        self.assertTrue(decision.detect_help_intent("我不懂"))

    def test_zero_baseline_triggers(self):
        self.assertTrue(decision.detect_help_intent("这个我没学过，零基础"))

    def test_advance_not_help(self):
        # 推进意图不应误判为求助
        self.assertFalse(decision.detect_help_intent("我懂了，继续下一个"))

    def test_normal_statement_not_help(self):
        # 正常理解陈述（长文本且非强信号）不应触发求助
        self.assertFalse(
            decision.detect_help_intent("HTTP 是无状态的应用层协议，用于客户端和服务器通信")
        )


class TestProcessTurnHelpIntent(unittest.TestCase):
    def test_help_intent_forces_explain(self):
        s = _make_session()
        ctx = main._process_turn(s, "我不懂")
        self.assertEqual(ctx["action"], "explain")
        self.assertTrue(ctx["help_intent"])
        self.assertFalse(ctx["should_advance"])

    def test_help_intent_diagnosis_stays_honest(self):
        # 诊断层不被篡改：用户说「我不懂」仍判 insufficient（记忆点诚实），
        # 且不把任何概念标记为 mastered（不假完成）。
        s = _make_session()
        ctx = main._process_turn(s, "我还不懂，能解释一下吗？")
        self.assertNotEqual(ctx["diagnosis"].state, "understood")
        self.assertFalse(ctx["should_advance"])
        self.assertFalse(any(m["mastered"] for m in ctx["cognitive"]["concepts"]))


class TestDagOverview(unittest.TestCase):
    def test_overview_contains_concepts_and_order(self):
        s = _make_session()
        knowledge = s["knowledge"]
        focus_id = knowledge["concepts"][0]["id"]
        text = tutor._dag_overview(knowledge, focus_id)
        self.assertIn("学习目标", text)
        self.assertIn(knowledge["concepts"][0]["name"], text)
        self.assertIn("当前正在学", text)

    def test_overview_empty_knowledge(self):
        self.assertEqual(tutor._dag_overview(None), "")


class TestExplainInjectsOverview(unittest.TestCase):
    def _diag(self, focus_id: str) -> DiagnosticResult:
        return DiagnosticResult(
            state="insufficient", confidence=0.8, concept_ids=[focus_id],
            evidence="", misconception="", missing=[], quality="",
        )

    def test_explain_prompt_contains_overview(self):
        s = _make_session()
        knowledge = s["knowledge"]
        concept = Concept(**knowledge["concepts"][0])
        sys_prompt = tutor._tutor_system(
            concept, self._diag(concept.id), "explain", knowledge
        )
        self.assertIn("领域全景", sys_prompt)
        self.assertIn(concept.name, sys_prompt)

    def test_probe_prompt_no_overview(self):
        # 非 explain 动作不注入全景（避免打扰「先问后教」的正常追问）
        s = _make_session()
        knowledge = s["knowledge"]
        concept = Concept(**knowledge["concepts"][0])
        sys_prompt = tutor._tutor_system(
            concept, self._diag(concept.id), "probe", knowledge
        )
        self.assertNotIn("领域全景", sys_prompt)


class TestBuildIntroNovice(unittest.TestCase):
    def test_novice_intro_has_overview(self):
        km = domain_model.build_knowledge_model("零基础入门 HTTP")
        knowledge = km.model_dump()
        focus = knowledge["concepts"][0]
        intro = tutor.build_intro(knowledge, focus)
        self.assertIn("地图", intro)
        self.assertIn("我不懂", intro)

    def test_non_novice_intro_asks(self):
        km = domain_model.build_knowledge_model("深入理解 HTTP 缓存原理")
        knowledge = km.model_dump()
        focus = knowledge["concepts"][0]
        intro = tutor.build_intro(knowledge, focus)
        self.assertIn("你能用自己的话说说", intro)


if __name__ == "__main__":
    unittest.main()
