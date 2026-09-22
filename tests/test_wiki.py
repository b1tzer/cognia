"""个人 Wiki 数据层（wiki_repo + wiki）单元测试。

用临时目录承载 Git 仓库（monkeypatch WIKI_REPO_ROOT）+ InMemoryStore（不依赖 Postgres），验证：
1. Git 版本层的写/读/历史/回滚
2. 领域逻辑的 CRUD + 溯源 + 多 user 隔离
3. store 降级不抛错、非法 page_id 拦截
"""

import pytest
from langgraph.store.memory import InMemoryStore

from cognia import wiki, wiki_repo
from cognia.schemas import WikiAuthor, WikiPage


@pytest.fixture
def repo_root(tmp_path, monkeypatch):
    root = tmp_path / "wiki"
    monkeypatch.setenv("WIKI_REPO_ROOT", str(root))
    return root


def test_wiki_repo_roundtrip(repo_root):
    """Git 版本层：写/读/历史/回滚闭环。"""
    uid = "user-a"
    wiki_repo.ensure_repo(uid)

    h1 = wiki_repo.write_page(uid, "spring-aop", "# AOP\n内容", "ai", "create")
    assert wiki_repo.read_page(uid, "spring-aop") == "# AOP\n内容"

    h2 = wiki_repo.write_page(uid, "spring-aop", "# AOP\n更新内容", "user", "update")
    assert h2 != h1

    hist = wiki_repo.history(uid, "spring-aop")
    assert len(hist) == 2
    assert hist[0]["commit_hash"] == h2  # 最新在前
    assert hist[0]["author"] == "user"
    assert hist[1]["author"] == "ai"

    h3 = wiki_repo.rollback(uid, "spring-aop", h1, "user")
    assert wiki_repo.read_page(uid, "spring-aop") == "# AOP\n内容"
    assert h3 != h2

    # 删除
    wiki_repo.delete_page(uid, "spring-aop", "user")
    assert wiki_repo.read_page(uid, "spring-aop") is None


def test_wiki_repo_commit_noop(repo_root):
    """内容未变时 commit 无变更，返回当前 HEAD 不算错误。"""
    uid = "user-a"
    h1 = wiki_repo.write_page(uid, "a", "内容", "ai")
    h2 = wiki_repo.write_page(uid, "a", "内容", "ai")  # 内容未变
    assert h1 == h2


def test_wiki_crud_and_isolation(repo_root):
    """领域逻辑：CRUD + 溯源字段 + 多 user 隔离。"""
    store = InMemoryStore()
    page = WikiPage(
        page_id="aop", title="AOP", content_markdown="# AOP",
        author=WikiAuthor.AI, source_thread_id="t1",
        source_turns=[1, 3], evidence=["用户说：切面就是..."],
    )
    meta = wiki.create_page(store, "u1", page)
    assert meta["commit_hash"]
    assert meta["source_thread_id"] == "t1"
    assert meta["source_turns"] == [1, 3]
    assert meta["author"] == "ai"

    got = wiki.get_page(store, "u1", "aop")
    assert got["content_markdown"] == "# AOP"
    assert got["evidence"] == ["用户说：切面就是..."]

    assert len(wiki.list_pages(store, "u1")) == 1
    # 隔离
    assert wiki.get_page(store, "u2", "aop") is None
    assert wiki.list_pages(store, "u2") == []

    # 更新（作者转 user）
    page2 = WikiPage(page_id="aop", title="AOP v2", content_markdown="# AOP v2", author=WikiAuthor.USER)
    meta2 = wiki.update_page(store, "u1", "aop", page2)
    assert meta2["author"] == "user"
    assert meta2["commit_hash"] != meta["commit_hash"]
    assert len(wiki.history("u1", "aop")) == 2

    # 删除
    assert wiki.delete_page(store, "u1", "aop") is True
    assert wiki.get_page(store, "u1", "aop") is None


def test_wiki_store_none_degradation(repo_root):
    """store 为 None：读降级返回空，写降级返回 None（不抛 500）。"""
    assert wiki.list_pages(None, "u1") == []
    assert wiki.get_page(None, "u1", "aop") is None
    assert wiki.create_page(None, "u1", WikiPage(page_id="a", title="a")) is None
    assert wiki.update_page(None, "u1", "a", WikiPage(page_id="a", title="a")) is None
    assert wiki.delete_page(None, "u1", "a") is False


def test_wiki_invalid_page_id(repo_root):
    """非法 page_id（路径穿越）被拦截。"""
    store = InMemoryStore()
    with pytest.raises(ValueError):
        wiki.create_page(store, "u1", WikiPage(page_id="../../etc/passwd", title="x"))
