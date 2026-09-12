"""全局概念库（Cognitive Atlas）归一化与数据模型（#31）的单元测试。

覆盖 normalize_key（全半角/空白/后缀/英文小写）、resolve_global_id（跨会话稳定匹配）、
db 概念/关系表读写、以及 domain_model 归一化集成（concept id 稳定）。
"""
from __future__ import annotations

import os
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


if __name__ == "__main__":
    unittest.main()
