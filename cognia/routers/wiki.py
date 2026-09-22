"""个人 Wiki 路由（/wiki CRUD + history + rollback）。

与 AG-UI 接入层解耦，属纯业务端点。正文存 Git、元数据存 Store（见 cognia/wiki.py）。
写操作经 ``asyncio.to_thread`` 包装同步领域逻辑（含 Git subprocess），避免阻塞事件循环。

安全边界：
- user_id 缺失 → 400。
- store 为 None（Postgres 降级）→ 读降级返回空，写返回 503（不 500、不静默）。
- page_id 非法 → 400；页面不存在 → 404；Git 异常 → 503。
"""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from cognia import wiki
from cognia.schemas import WikiAuthor, WikiPage
from cognia.wiki_repo import WikiRepoError

router = APIRouter()


class WikiUpsertRequest(BaseModel):
    """新建 / 更新页面的请求体（正文 + 溯源字段，不含 updated_at）。"""

    page_id: str
    title: str
    content_markdown: str = ""
    path: str = ""
    tags: list[str] = []
    parent_page_id: str | None = None
    author: str = WikiAuthor.USER.value
    source_thread_id: str | None = None
    source_turns: list[int] = []
    evidence: list[str] = []


class RollbackRequest(BaseModel):
    """回滚请求体。"""

    commit_hash: str


def _store(request: Request):
    return getattr(request.app.state, "store", None)


def _require_store(request: Request):
    """写操作前置：store 缺失（Postgres 降级）时拒绝，避免静默丢数据。"""
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="长期记忆未启用，wiki 不可写")
    return store


def _require_user(user_id: str | None) -> str:
    if not user_id:
        raise HTTPException(status_code=400, detail="缺少 user_id 参数")
    return user_id


def _to_page(req: WikiUpsertRequest) -> WikiPage:
    try:
        author = WikiAuthor(req.author)
    except ValueError:
        author = WikiAuthor.USER
    return WikiPage(
        page_id=req.page_id,
        title=req.title,
        content_markdown=req.content_markdown,
        path=req.path,
        tags=req.tags,
        parent_page_id=req.parent_page_id,
        author=author,
        source_thread_id=req.source_thread_id,
        source_turns=req.source_turns,
        evidence=req.evidence,
    )


async def _call(fn, *args):
    """把同步领域逻辑丢到 executor 线程执行，统一翻译领域异常为 HTTP。"""
    try:
        return await asyncio.to_thread(fn, *args)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except WikiRepoError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/wiki")
async def list_wiki(request: Request, user_id: str | None = None):
    """列出某 user 全部 wiki 页面元数据（不含正文），按 updated_at 降序。"""
    _require_user(user_id)
    pages = await _call(wiki.list_pages, _store(request), user_id)
    return {"pages": pages}


@router.get("/wiki/{page_id}")
async def get_wiki(request: Request, page_id: str, user_id: str | None = None):
    """读单页完整内容（元数据 + 正文）。"""
    _require_user(user_id)
    page = await _call(wiki.get_page, _store(request), user_id, page_id)
    if page is None:
        raise HTTPException(status_code=404, detail="页面不存在")
    return page


@router.post("/wiki")
async def create_wiki(request: Request, req: WikiUpsertRequest, user_id: str | None = None):
    """新建一页（写 Git + 写元数据）。"""
    _require_user(user_id)
    store = _require_store(request)
    return await _call(wiki.create_page, store, user_id, _to_page(req))


@router.put("/wiki/{page_id}")
async def update_wiki(
    request: Request, page_id: str, req: WikiUpsertRequest, user_id: str | None = None,
):
    """更新一页（写新版本 + 更新元数据）。"""
    _require_user(user_id)
    store = _require_store(request)
    return await _call(wiki.update_page, store, user_id, page_id, _to_page(req))


@router.delete("/wiki/{page_id}")
async def delete_wiki(request: Request, page_id: str, user_id: str | None = None):
    """删除一页（删 Git + 删元数据）。"""
    _require_user(user_id)
    store = _require_store(request)
    deleted = await _call(wiki.delete_page, store, user_id, page_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="页面不存在")
    return {"deleted": True, "page_id": page_id}


@router.get("/wiki/{page_id}/history")
async def wiki_history(request: Request, page_id: str, user_id: str | None = None):
    """读某页版本历史（git log，最新在前）。"""
    _require_user(user_id)
    items = await _call(wiki.history, user_id, page_id)
    return {"page_id": page_id, "revisions": items}


@router.post("/wiki/{page_id}/rollback")
async def wiki_rollback(
    request: Request, page_id: str, req: RollbackRequest, user_id: str | None = None,
):
    """回滚一页到指定 commit，产生新版本。"""
    _require_user(user_id)
    store = _require_store(request)
    return await _call(
        wiki.rollback, store, user_id, page_id, req.commit_hash, WikiAuthor.USER.value,
    )
