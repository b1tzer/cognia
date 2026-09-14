"""需求 #71 需求4：开场/推进提问措辞去模板化的测试。

覆盖：
1. tutor.build_transition 过渡提问（AI 关闭降级模板，含下一概念名）
2. tutor._intro_with_llm 开场白 LLM 生成（AI 关闭返回 None，降级）
3. build_intro 去模板化：AI 关闭时降级到模板；LLM 提示词要求不套固定句式

强制 AI_ENABLED=False 走确定性降级路径，保证可复现、不依赖外部模型。
"""
from __future__ import annotations

import unittest

import config

config.AI_ENABLED = False

import domain_model
import tutor


def _make_knowledge(goal: str = "理解 HTTP 缓存原理") -> tuple:
    km = domain_model.build_knowledge_model(goal)
    knowledge = km.model_dump()
    focus = knowledge["concepts"][0]
    return knowledge, focus


class TestBuildTransition(unittest.TestCase):
    def test_transition_template_contains_next_concept(self):
        knowledge, focus = _make_knowledge()
        text = tutor.build_transition(focus, knowledge)
        # 降级模板应包含下一概念名（承接上下文）
        self.assertIn(focus["name"], text)

    def test_transition_template_mentions_advance(self):
        knowledge, focus = _make_knowledge()
        text = tutor.build_transition(focus, knowledge)
        # 降级模板应体现「已掌握、接着看下一个」的推进语义
        self.assertIn("下一个", text)

    def test_transition_no_knowledge(self):
        # knowledge 缺省时也能降级（不因领域地图缺失而崩溃）
        _, focus = _make_knowledge()
        text = tutor.build_transition(focus, None)
        self.assertIn(focus["name"], text)


class TestIntroDetemplating(unittest.TestCase):
    def test_intro_with_llm_offline_returns_none(self):
        knowledge, focus = _make_knowledge()
        # AI 关闭时 LLM 生成路径应返回 None，由 build_intro 降级到模板
        self.assertIsNone(tutor._intro_with_llm(knowledge, focus))

    def test_intro_system_forbids_fixed_phrasing(self):
        # LLM 提示词应明确禁止「X 是什么、解决什么问题」固定句式（去模板化核心）
        self.assertIn("严禁出现", tutor._INTRO_SYSTEM)
        self.assertIn("是什么、解决什么问题", tutor._INTRO_SYSTEM)

    def test_transition_system_forbids_fixed_phrasing(self):
        self.assertIn("严禁出现", tutor._TRANSITION_SYSTEM)
        self.assertIn("是什么、解决什么问题", tutor._TRANSITION_SYSTEM)


if __name__ == "__main__":
    unittest.main()
