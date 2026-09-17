"""cognia.concept_merge 概念语义合并引擎单元测试。

纯函数、零 LLM、零 IO。embedding 用 monkeypatch 打桩（不依赖真实模型下载），
覆盖四类场景：
1. 同 name 合并（embedding 不可用时的降级路径）
2. 同义不同名合并（embedding 可用、cosine 高）
3. 不同义不合并
4. 全新概念 / 空存量
"""

from cognia import concept_merge


def test_same_name_merge_without_embedding(monkeypatch):
    # embedding 不可用 → 退化为归一化 name 精确匹配（同名才合并）
    monkeypatch.setattr("cognia.embedding.embed_text", lambda t: None)
    existing = [{"id": "concept:aaa", "name": "Spring AOP", "embedding": None}]

    gid, is_new, matched = concept_merge.merge_concept("spring  aop", existing)

    assert is_new is False
    assert gid == "concept:aaa"
    assert matched == "Spring AOP"


def test_synonym_merge_with_embedding(monkeypatch):
    # embedding 可用：同义不同名，cosine 高 → 合并
    monkeypatch.setattr("cognia.embedding.embed_text", lambda t: [1.0])
    monkeypatch.setattr("cognia.embedding.cosine", lambda a, b: 0.92)
    existing = [{"id": "concept:aaa", "name": "Spring AOP", "embedding": [1.0]}]

    gid, is_new, _ = concept_merge.merge_concept("切面编程", existing)

    assert is_new is False
    assert gid == "concept:aaa"


def test_different_concept_not_merge(monkeypatch):
    # 不同义：cosine 低于阈值 → 不合并，新建
    monkeypatch.setattr("cognia.embedding.embed_text", lambda t: [1.0])
    monkeypatch.setattr("cognia.embedding.cosine", lambda a, b: 0.30)
    existing = [{"id": "concept:aaa", "name": "线程", "embedding": [1.0]}]

    gid, is_new, _ = concept_merge.merge_concept("进程", existing)

    assert is_new is True
    assert gid == concept_merge.concept_id("进程")


def test_new_concept_empty_existing(monkeypatch):
    monkeypatch.setattr("cognia.embedding.embed_text", lambda t: None)
    gid, is_new, _ = concept_merge.merge_concept("Spring AOP", [])

    assert is_new is True
    assert gid == concept_merge.concept_id("Spring AOP")


def test_concept_id_stable():
    # 同一 name 归一化后 id 稳定，不同 name 不同 id
    a = concept_merge.concept_id("Spring AOP")
    b = concept_merge.concept_id("spring  aop")
    c = concept_merge.concept_id("切面编程")
    assert a == b
    assert a != c
    assert a.startswith("concept:")
