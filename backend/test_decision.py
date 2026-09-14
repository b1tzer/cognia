"""decision.py 焦点概念选择引擎的单元测试。

覆盖：拓扑排序、ZPD 评分、焦点候选集、掌握判定、回溯检测。

用标准库 unittest，不引入额外依赖。强制 AI_ENABLED=False 走确定性规则路径，
保证测试可复现、不依赖外部模型。
"""
from __future__ import annotations

import unittest

import config
import decision

# 强制离线规则路径（保证测试确定性）
config.AI_ENABLED = False

# ---------------------------------------------------------------------------
# 测试数据：简单 DAG  a -> b -> c（root=c）
# ---------------------------------------------------------------------------
KNOWLEDGE = {
    "goal": "测试目标",
    "root_concepts": ["c"],
    "concepts": [
        {"id": "a", "name": "概念A", "summary": "", "why_matters": "",
         "prerequisites": [], "common_misconceptions": []},
        {"id": "b", "name": "概念B", "summary": "", "why_matters": "",
         "prerequisites": ["a"], "common_misconceptions": []},
        {"id": "c", "name": "概念C", "summary": "", "why_matters": "",
         "prerequisites": ["b"], "common_misconceptions": []},
    ],
}


def make_cognitive(
    masteries: dict,
    failures: dict | None = None,
    success_counts: dict | None = None,
    qualities: dict | None = None,
    evidence_counts: dict | None = None,
    mastered: dict | None = None,
) -> dict:
    """构造认知模型。

    masteries: {cid: mastery}
    failures: {cid: consecutive_failures}
    success_counts: {cid: success_count}（证据充分性）
    qualities: {cid: quality}（deep / surface / ""）
    evidence_counts: {cid: evidence_count}（已收集证据条数，默认 0=从未学过）
    mastered: {cid: bool}（是否已完成该概念学习，推进的权威标记）
    """
    failures = failures or {}
    success_counts = success_counts or {}
    qualities = qualities or {}
    evidence_counts = evidence_counts or {}
    mastered = mastered or {}
    names = {"a": "概念A", "b": "概念B", "c": "概念C"}
    return {
        "goal": "测试目标",
        "profile": {"level": "intermediate", "params": {}},
        "concepts": [
            {
                "concept_id": cid,
                "concept_name": names[cid],
                "mastery": masteries.get(cid, 0.0),
                "state": "insufficient",
                "evidence_count": evidence_counts.get(cid, 0),
                "consecutive_failures": failures.get(cid, 0),
                "last_evidence": "",
                "success_count": success_counts.get(cid, 0),
                "quality": qualities.get(cid, ""),
                "mastered": mastered.get(cid, False),
            }
            for cid in ("a", "b", "c")
        ],
        "updated_at": "",
    }


class TestTopoOrder(unittest.TestCase):
    def test_topo_order(self):
        self.assertEqual(decision.topo_order(KNOWLEDGE), ["a", "b", "c"])


class TestZpdScore(unittest.TestCase):
    def test_mastered_returns_zero(self):
        self.assertEqual(decision.zpd_score(0.85), 0.0)

    def test_in_zone_returns_one(self):
        self.assertEqual(decision.zpd_score(0.7), 1.0)

    def test_low_mastery_scaled(self):
        # 0.3 < ZPD_MIN(0.6): 0.3 + 0.7*(0.3/0.6) = 0.65
        self.assertAlmostEqual(decision.zpd_score(0.3), 0.65, places=6)

    def test_zero_mastery(self):
        self.assertAlmostEqual(decision.zpd_score(0.0), 0.3, places=6)


class TestFocusCandidates(unittest.TestCase):
    def test_build_focus_candidates(self):
        # a 已 mastered，b 未掌握且前置 a 已 mastered → 候选只有 b；c 因前置 b 未 mastered 被排除
        cog = make_cognitive(
            {"a": 0.85, "b": 0.3, "c": 0.3},
            mastered={"a": True},
        )
        cands = decision.build_focus_candidates(KNOWLEDGE, cog)
        self.assertEqual([c["concept"]["id"] for c in cands], ["b"])

    def test_next_focus_concept_rule(self):
        cog = make_cognitive(
            {"a": 0.85, "b": 0.3, "c": 0.3},
            mastered={"a": True},
        )
        focus = decision.next_focus_concept(KNOWLEDGE, cog)
        self.assertEqual(focus["id"], "b")

    def test_next_focus_concept_all_mastered_returns_none(self):
        cog = make_cognitive(
            {"a": 0.85, "b": 0.85, "c": 0.85},
            mastered={"a": True, "b": True, "c": True},
        )
        self.assertIsNone(decision.next_focus_concept(KNOWLEDGE, cog))


class TestIsMastered(unittest.TestCase):
    """is_mastered 单一 mastered 布尔判定（推进的权威标记）。"""

    def _m(self, mastered=False):
        return {"mastered": mastered}

    def test_mastered_true_returns_true(self):
        self.assertTrue(decision.is_mastered(self._m(True)))

    def test_mastered_false_returns_false(self):
        self.assertFalse(decision.is_mastered(self._m(False)))

    def test_legacy_numeric_fields_no_longer_matter(self):
        # 旧三层数值字段不再影响掌握判定（即使 mastery 高 / success_count 够 / quality deep）
        m = {"mastery": 0.99, "success_count": 3, "quality": "deep", "mastered": False}
        self.assertFalse(decision.is_mastered(m))

    def test_missing_fields_fallback(self):
        # 旧数据缺字段 → get 默认值 False，不判掌握，不抛异常
        self.assertFalse(decision.is_mastered({}))


class TestBacktrack(unittest.TestCase):
    def test_weakest_prereq(self):
        mastery_map = {
            "a": {"mastery": 0.85},
            "b": {"mastery": 0.3},
            "c": {"mastery": 0.3},
        }
        concepts = {c["id"]: c for c in KNOWLEDGE["concepts"]}
        self.assertEqual(decision.weakest_prereq("b", concepts, mastery_map), "a")

    def test_detect_backtrack_consecutive_failure(self):
        # b 连续失败 3 次 → 回溯到其最弱前置 a
        cog = make_cognitive({"a": 0.85, "b": 0.3, "c": 0.3}, failures={"b": 3})
        self.assertEqual(decision.detect_backtrack_target(KNOWLEDGE, cog), "a")

    def test_detect_backtrack_prereq_decay(self):
        # b 的前置 a「学过但衰退」到 0.4（< floor 0.5）→ 回溯到 a
        cog = make_cognitive({"a": 0.4, "b": 0.3, "c": 0.3}, evidence_counts={"a": 3})
        self.assertEqual(decision.detect_backtrack_target(KNOWLEDGE, cog), "a")

    def test_detect_backtrack_unlearned_prereq_not_decay(self):
        # b 的前置 a「从未学过」（evidence_count=0，mastery 为初始值 0.35 < floor 0.5）
        # 不应被误判为「衰退」而触发回溯（回归：修复前会误返回 "a"）
        cog = make_cognitive({"a": 0.35, "b": 0.35, "c": 0.35})
        self.assertIsNone(decision.detect_backtrack_target(KNOWLEDGE, cog))

    def test_next_focus_concept_backtrack_priority(self):
        # 回溯优先于正常候选：b 连续失败 → 焦点切回 a（即使 a 已掌握也回去巩固）
        cog = make_cognitive({"a": 0.85, "b": 0.3, "c": 0.3}, failures={"b": 3})
        focus = decision.next_focus_concept(KNOWLEDGE, cog)
        self.assertEqual(focus["id"], "a")


if __name__ == "__main__":
    unittest.main()
