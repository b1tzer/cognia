"""个人 Wiki 的领域存储逻辑（Store 元数据 + Git 正文的编排）。

分工：正文（content_markdown）存 Git（版本真相，见 wiki_repo.py），元数据 +
溯源（title/tags/parent/author/source_thread_id/evidence/commit_hash）存
Postgres Store 的 ``("wiki", user_id)`` namespace，便于列表 / 溯源的结构化查询。

本模块逻辑函数接受 ``store`` 参数、不自行建连接（测试用 InMemoryStore 注入）；
Git 写读经 ``wiki_repo`` 完成。写操作内部先写 Git（拿 commit_hash）再写 Store 元数据。

安全边界：
- store 为 None（Postgres 降级）→ 读降级返回空 / None，写返回 None（由 router 转 503），不抛 500。
- page_id 非法（防路径穿越）→ 抛 ValueError（router 转 400）。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from langgraph.store.base import BaseStore

from cognia.schemas import WikiAuthor, WikiPage
from cognia import wiki_repo

WIKI_NS = "wiki"

# page_id 只允许字母数字开头、后接字母数字/下划线/连字符，防路径穿越。
_PAGE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")


def _check_page_id(page_id: str) -> None:
    if not _PAGE_ID_RE.match(page_id or ""):
        raise ValueError(f"非法 page_id：{page_id}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _meta_value(page: WikiPage, commit_hash: str) -> dict:
    """由 WikiPage 生成元数据 dict（剔除正文，附加 commit_hash）。"""
    meta = page.model_dump(mode="json")
    meta.pop("content_markdown", None)  # 正文存 Git，不入 Store
    meta["commit_hash"] = commit_hash
    return meta


def _sort_by_updated_desc(items) -> list[dict]:
    values = [item.value for item in items if item.value is not None]
    values.sort(key=lambda d: d.get("updated_at") or "", reverse=True)
    return values


# ---- 读 ----

def list_pages(store: BaseStore | None, user_id: str) -> list[dict]:
    """列出某 user 全部 wiki 页面元数据（不含正文），按 updated_at 降序。"""
    if store is None or not user_id:
        return []
    return _sort_by_updated_desc(store.search((WIKI_NS, user_id)))


def get_page_meta(store: BaseStore | None, user_id: str, page_id: str) -> dict | None:
    """读某页元数据（不含正文），不存在返回 None。"""
    if store is None or not user_id:
        return None
    item = store.get((WIKI_NS, user_id), page_id)
    return item.value if item else None


def get_page(store: BaseStore | None, user_id: str, page_id: str) -> dict | None:
    """读某页完整内容（元数据 + Git 正文），不存在返回 None。"""
    meta = get_page_meta(store, user_id, page_id)
    if meta is None:
        return None
    content = wiki_repo.read_page(user_id, page_id)
    meta["content_markdown"] = content or ""
    return meta


def history(user_id: str, page_id: str) -> list[dict]:
    """读某页版本历史（纯 Git，不依赖 store）。"""
    return wiki_repo.history(user_id, page_id)


# ---- 写 ----

def create_page(store: BaseStore | None, user_id: str, page: WikiPage) -> dict | None:
    """新建一页：写 Git 正文 + 写 Store 元数据，返回元数据（含 commit_hash）。

    store 为 None → 返回 None（未持久化，router 转 503）。
    """
    if store is None or not user_id:
        return None
    _check_page_id(page.page_id)
    page.updated_at = datetime.now(timezone.utc)
    commit = wiki_repo.write_page(
        user_id, page.page_id, page.content_markdown, page.author.value,
        summary=f"create {page.page_id}",
    )
    meta = _meta_value(page, commit)
    store.put((WIKI_NS, user_id), page.page_id, meta)
    return meta


def update_page(
    store: BaseStore | None, user_id: str, page_id: str, page: WikiPage,
) -> dict | None:
    """更新一页：写 Git 正文（新版本）+ 更新 Store 元数据，返回元数据。"""
    if store is None or not user_id:
        return None
    _check_page_id(page_id)
    if get_page_meta(store, user_id, page_id) is None:
        raise KeyError(f"页面不存在：{page_id}")
    page.page_id = page_id
    page.updated_at = datetime.now(timezone.utc)
    commit = wiki_repo.write_page(
        user_id, page_id, page.content_markdown, page.author.value,
        summary=f"update {page_id}",
    )
    meta = _meta_value(page, commit)
    store.put((WIKI_NS, user_id), page_id, meta)
    return meta


def delete_page(store: BaseStore | None, user_id: str, page_id: str) -> bool:
    """删除一页：删 Git 正文 + 删 Store 元数据，返回是否真的删除。"""
    if store is None or not user_id:
        return False
    _check_page_id(page_id)
    if get_page_meta(store, user_id, page_id) is None:
        return False
    wiki_repo.delete_page(user_id, page_id, WikiAuthor.USER.value)
    store.delete((WIKI_NS, user_id), page_id)
    return True


def rollback(
    store: BaseStore | None, user_id: str, page_id: str, commit_hash: str,
    author: str = WikiAuthor.USER.value,
) -> dict | None:
    """回滚一页到指定 commit：Git 回滚 + 更新 Store 元数据，返回元数据。"""
    if store is None or not user_id:
        return None
    _check_page_id(page_id)
    meta = get_page_meta(store, user_id, page_id)
    if meta is None:
        raise KeyError(f"页面不存在：{page_id}")
    new_commit = wiki_repo.rollback(user_id, page_id, commit_hash, author)
    meta["commit_hash"] = new_commit
    meta["author"] = author
    meta["updated_at"] = _now_iso()
    store.put((WIKI_NS, user_id), page_id, meta)
    return meta
