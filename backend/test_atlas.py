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


if __name__ == "__main__":
    unittest.main()
