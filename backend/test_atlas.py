"""全局概念库（Cognitive Atlas）归一化与数据模型（#31）的单元测试。

覆盖 normalize_key（全半角/空白/后缀/英文小写）、resolve_global_id（跨会话稳定匹配）、
db 概念/关系表读写、以及 domain_model 归一化集成（concept id 稳定）。
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

import config

config.AI_ENABLED = False
config.DB_PATH = os.path.join(tempfile.mkdtemp(), "atlas_test.db")

import db  # noqa: E402
import atlas  # noqa: E402
import domain_model  # noqa: E402


class TestNormalizeKey(unittest.TestCase):
    def test_strips_generic_suffix(self):
        self.assertEqual(atlas.normalize_key("HTTP 基础"), "http")
        self.assertEqual(atlas.normalize_key("HTTP 协议"), "http")

    def test_full_to_half_and_whitespace(self):
        self.assertEqual(atlas.normalize_key("ＨＴＴＰ　协议"), "http")
        self.assertEqual(atlas.normalize_key("  并发 机制  "), "并发")

    def test_english_lowercase(self):
        self.assertEqual(atlas.normalize_key("TLS"), "tls")

    def test_multi_suffix_loop(self):
        self.assertEqual(atlas.normalize_key("HTTP 协议基础"), "http")

    def test_empty_returns_empty(self):
        self.assertEqual(atlas.normalize_key(""), "")
        self.assertEqual(atlas.normalize_key("  "), "")


class TestResolveGlobalId(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def test_same_name_same_id(self):
        a = atlas.resolve_global_id("HTTP 基础", "s")
        b = atlas.resolve_global_id("HTTP 协议", "s")
        self.assertEqual(a, b)
        self.assertEqual(a, "http")

    def test_different_name_different_id(self):
        a = atlas.resolve_global_id("并发", "s")
        b = atlas.resolve_global_id("缓存", "s")
        self.assertNotEqual(a, b)

    def test_empty_name_fallback_unique(self):
        cid = atlas.resolve_global_id("")
        self.assertTrue(cid.startswith("c-"))


class TestConceptTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def test_upsert_and_get(self):
        db.upsert_concept("tls", "TLS", "transport security")
        row = db.get_concept("tls")
        self.assertIsNotNone(row)
        self.assertEqual(row["name"], "TLS")

    def test_list_concepts(self):
        db.upsert_concept("zzz-test", "ZZZ Test")
        ids = [c["id"] for c in db.list_concepts()]
        self.assertIn("zzz-test", ids)

    def test_upsert_is_idempotent(self):
        db.upsert_concept("idem", "name1")
        db.upsert_concept("idem", "name2")
        self.assertEqual(db.get_concept("idem")["name"], "name2")


class TestConceptRelation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def test_upsert_and_list(self):
        db.upsert_concept_relation("a", "b", "is-a")
        db.upsert_concept_relation("a", "b", "is-a")  # 幂等去重
        rels = db.list_concept_relations()
        # 相同 (from,to,type) 只保留一条
        self.assertEqual(
            sum(1 for r in rels if r["from_id"] == "a" and r["to_id"] == "b" and r["relation_type"] == "is-a"),
            1,
        )

    def test_self_loop_ignored(self):
        db.upsert_concept_relation("x", "x", "related")
        self.assertEqual(db.get_concept_relations("x"), [])

    def test_get_by_concept(self):
        db.upsert_concept_relation("from-t", "to-t", "prerequisite")
        rels = db.get_concept_relations("from-t")
        self.assertTrue(any(r["to_id"] == "to-t" for r in rels))


class TestDomainModelNormalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        db.init_db()

    def test_concept_ids_are_stable_global_ids(self):
        km = domain_model.build_knowledge_model("学习 HTTP")
        ids = [c.id for c in km.concepts]
        # 模板原始 id 不再出现（已被归一化为全局稳定 id）
        self.assertNotIn("http-basics", ids)
        self.assertIn("http", ids)
        # prerequisites 引用应被同步替换，指向已存在的概念 id
        for c in km.concepts:
            for p in c.prerequisites:
                self.assertIn(p, ids)

    def test_rebuild_same_ids(self):
        km1 = domain_model.build_knowledge_model("学习 HTTP")
        km2 = domain_model.build_knowledge_model("学习 HTTP")
        self.assertEqual([c.id for c in km1.concepts], [c.id for c in km2.concepts])


class TestBuildAtlasView(unittest.TestCase):
    def setUp(self):
        db.init_db()
        # 清空全局概念库 / 关系 / 掌握度，保证空库与聚合场景可独立验证
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute("DELETE FROM concepts")
        conn.execute("DELETE FROM concept_relations")
        conn.execute("DELETE FROM concept_mastery")
        conn.commit()
        conn.close()

    def test_empty_atlas_returns_empty_lists(self):
        view = atlas.build_atlas_view("nobody")
        self.assertEqual(view["concepts"], [])
        self.assertEqual(view["relations"], [])

    def test_aggregates_concepts_and_mastery(self):
        atlas.resolve_global_id("并发", "s")
        atlas.resolve_global_id("缓存", "s")
        db.upsert_concept_mastery("atlas-u1", "并发", 0.9)
        db.upsert_concept_relation("并发", "缓存", "related")

        view = atlas.build_atlas_view("atlas-u1")
        ids = [c["id"] for c in view["concepts"]]
        self.assertIn("并发", ids)
        self.assertIn("缓存", ids)
        for c in view["concepts"]:
            if c["id"] == "并发":
                self.assertGreater(c["mastery"], 0.89)
                self.assertEqual(c["state"], "understood")
            if c["id"] == "缓存":
                self.assertEqual(c["mastery"], 0.0)
                self.assertEqual(c["state"], "insufficient")
        self.assertTrue(any(
            r["from"] == "并发" and r["to"] == "缓存" and r["relation_type"] == "related"
            for r in view["relations"]
        ))


class TestNeighbors(unittest.TestCase):
    def setUp(self):
        db.init_db()
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute("DELETE FROM concepts")
        conn.execute("DELETE FROM concept_relations")
        conn.commit()
        conn.close()

    def test_isolated_concept_empty(self):
        self.assertEqual(atlas.neighbors("lonely", 1), [])

    def test_one_hop_neighbors(self):
        db.upsert_concept_relation("a", "b", "prerequisite")
        db.upsert_concept_relation("a", "c", "related")
        ns = atlas.neighbors("a", 1)
        self.assertEqual(len(ns), 2)
        self.assertTrue(any(n["node"] == "b" and n["relation_type"] == "prerequisite" for n in ns))
        self.assertTrue(any(n["node"] == "c" and n["relation_type"] == "related" for n in ns))

    def test_two_hop_depth(self):
        db.upsert_concept_relation("a", "b", "is-a")
        db.upsert_concept_relation("b", "c", "is-a")
        ns = atlas.neighbors("a", 2)
        nodes = {n["node"] for n in ns}
        self.assertIn("b", nodes)
        self.assertIn("c", nodes)

    def test_depth_less_than_one_defaults_to_one(self):
        db.upsert_concept_relation("a", "b", "related")
        ns = atlas.neighbors("a", 0)
        self.assertEqual(len(ns), 1)


class TestBridgeDetection(unittest.TestCase):
    def test_betweenness_line_graph(self):
        # a-b-c 线形图，b 的介数中心性最高
        ids = ["a", "b", "c"]
        rels = [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}]
        bc = atlas.betweenness_centrality(ids, rels)
        self.assertGreater(bc["b"], bc["a"])
        self.assertGreater(bc["b"], bc["c"])

    def test_find_bridge_two_mastered_neighbors(self):
        ids = ["a", "b", "c"]
        rels = [
            {"from": "a", "to": "b", "relation_type": "related"},
            {"from": "b", "to": "c", "relation_type": "related"},
        ]
        mastery = {"a": 0.9, "b": 0.1, "c": 0.9}
        bridges = atlas.find_bridge_paths(ids, rels, mastery)
        self.assertEqual(len(bridges), 1)
        self.assertEqual(bridges[0]["bridge"], "b")
        self.assertEqual(set(bridges[0]["neighbors"]), {"a", "c"})

    def test_find_bridge_no_bridge_when_all_mastered(self):
        ids = ["a", "b", "c"]
        rels = [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}]
        mastery = {"a": 0.9, "b": 0.9, "c": 0.9}
        bridges = atlas.find_bridge_paths(ids, rels, mastery)
        self.assertEqual(bridges, [])

    def test_find_bridge_empty_graph(self):
        self.assertEqual(atlas.find_bridge_paths([], [], {}), [])


class TestBuildKnowledgeFromConcept(unittest.TestCase):
    def setUp(self):
        db.init_db()
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute("DELETE FROM concepts")
        conn.execute("DELETE FROM concept_relations")
        conn.commit()
        conn.close()

    def test_missing_concept_returns_none(self):
        self.assertIsNone(atlas.build_knowledge_from_concept("not-exist"))

    def test_builds_subgraph(self):
        db.upsert_concept("target", "目标概念", "s")
        db.upsert_concept("pre", "前置概念", "s")
        db.upsert_concept("rel", "相关概念", "s")
        db.upsert_concept_relation("pre", "target", "prerequisite")
        db.upsert_concept_relation("target", "rel", "related")

        km = atlas.build_knowledge_from_concept("target")
        self.assertIsNotNone(km)
        self.assertEqual(km["goal"], "目标概念")
        ids = {c["id"] for c in km["concepts"]}
        self.assertIn("target", ids)
        self.assertIn("pre", ids)
        self.assertIn("rel", ids)
        target = next(c for c in km["concepts"] if c["id"] == "target")
        self.assertIn("pre", target["prerequisites"])

    def test_single_concept_no_relations(self):
        db.upsert_concept("solo", "孤立概念", "s")
        km = atlas.build_knowledge_from_concept("solo")
        self.assertEqual(len(km["concepts"]), 1)
        self.assertEqual(km["root_concepts"], ["solo"])


class TestLiveMastery(unittest.TestCase):
    def setUp(self):
        db.init_db()
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute("DELETE FROM sessions")
        conn.commit()
        conn.close()

    def test_aggregates_active_session_mastery(self):
        sid = db.create_session("学习X")["id"]
        cognitive = {
            "goal": "学习X",
            "concepts": [
                {"concept_id": "a", "mastery": 0.7, "state": "partial"},
                {"concept_id": "b", "mastery": 0.9, "state": "understood"},
            ],
        }
        db.update_session(sid, cognitive, {"concepts": []}, status="active")
        live = atlas._live_mastery_from_active_sessions()
        self.assertIn("a", live)
        self.assertIn("b", live)
        self.assertAlmostEqual(live["a"], 0.7)
        self.assertAlmostEqual(live["b"], 0.9)

    def test_ignores_completed_sessions(self):
        sid = db.create_session("学习X")["id"]
        cognitive = {"goal": "学习X", "concepts": [{"concept_id": "a", "mastery": 0.7}]}
        db.update_session(sid, cognitive, {"concepts": []}, status="completed")
        live = atlas._live_mastery_from_active_sessions()
        self.assertNotIn("a", live)


class TestRelationPersistence(unittest.TestCase):
    """#42：概念关系数据生产（prerequisite 从 DAG 落库 + related 从 LLM 输出落库 + 幂等）。"""

    def setUp(self):
        db.init_db()
        conn = sqlite3.connect(config.DB_PATH)
        conn.execute("DELETE FROM concepts")
        conn.execute("DELETE FROM concept_relations")
        conn.commit()
        conn.close()

    def test_prerequisite_persisted_from_dag(self):
        km = domain_model.build_knowledge_model("学习 HTTP")
        rels = db.list_concept_relations()
        prereq = {(r["from_id"], r["to_id"]) for r in rels if r["relation_type"] == "prerequisite"}
        # 每个概念的每个前置都应有一条 prerequisite 关系，且 from=前置、to=后继
        for c in km.concepts:
            for p in c.prerequisites:
                self.assertIn((p, c.id), prereq)
        # HTTP 模板 DAG 非空，至少有一条 prerequisite 落库
        self.assertGreaterEqual(len(prereq), 1)

    def test_related_persisted_from_llm_output(self):
        from schemas import Concept, KnowledgeModel
        c1 = Concept(id="http", name="HTTP 基础")
        c2 = Concept(id="session", name="会话与状态保持", related=["Cookie"])
        km = KnowledgeModel(goal="x", root_concepts=["session"], concepts=[c1, c2])
        domain_model._persist_relations(km)
        rels = db.list_concept_relations()
        cookie_id = atlas.normalize_key("Cookie")
        self.assertTrue(any(
            r["from_id"] == "session" and r["to_id"] == cookie_id and r["relation_type"] == "related"
            for r in rels
        ))

    def test_persist_relations_idempotent(self):
        from schemas import Concept, KnowledgeModel
        c1 = Concept(id="a", name="A")
        c2 = Concept(id="b", name="B", prerequisites=["a"], related=["C"])
        km = KnowledgeModel(goal="x", root_concepts=["b"], concepts=[c1, c2])
        domain_model._persist_relations(km)
        n1 = len(db.list_concept_relations())
        domain_model._persist_relations(km)
        n2 = len(db.list_concept_relations())
        self.assertEqual(n1, n2)


if __name__ == "__main__":
    unittest.main()
