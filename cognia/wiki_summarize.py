"""个人 Wiki 的对话总结层：把一次学习会话提炼为 wiki 草稿。

核心函数 :func:`summarize_thread_to_wiki` 被两处复用：
1. ``POST /wiki/summarize`` 端点（前端「生成 wiki」按钮触发）。
2. ``summarize_session_to_wiki`` Agent tool（对话内 AI 可控调用）。

流程：读 checkpointer 对话历史 → 提取 user/assistant 文本 → LLM 提炼为
:class:`WikiSummary`（markdown + 溯源证据）→ 写 wiki 草稿（author=ai +
source_thread_id + evidence）。所有失败均安全降级为 ``{"ok": False, "reason": ...}``，
不抛异常到调用方。
"""

from __future__ import annotations

import re

from cognia import models, wiki
from cognia.schemas import WikiAuthor, WikiPage, WikiSummary

_SUMMARIZE_SYSTEM = (
    "你是 Cognia 的知识整理助手，负责把一段「AI 学习教练与学员」的对话总结为一篇"
    "体系化的 wiki 文章。\n\n"
    "要求：\n"
    "1. 用 markdown 输出正文，结构清晰（标题/小节/要点/代码块皆可），只保留技术事实"
    "与结论，剔除闲聊、客套与重复内容。\n"
    "2. evidence 字段：从【user】原话里摘取 2~5 句最能佐证学员理解程度/关键结论的"
    "原话，必须逐字引用、不得改写或脑补；学员原话过少时可少于 2 条。\n"
    "3. tags 字段：3~6 个英文小写标签。\n"
    "4. slug 字段：英文小写连字符（如 spring-aop-proxy）。\n"
    "5. title 字段：简洁中文标题。\n\n"
    "严格只输出一个 JSON 对象，不要输出任何解释或 markdown 代码块：\n"
    '{"title": "...", "slug": "...", "markdown": "...", "evidence": ["..."], "tags": ["..."]}'
)


def _extract_dialogue(messages) -> tuple[list[dict], list[int]]:
    """从 checkpoint messages 提取 user/assistant 文本对话 + 用户轮次序号。"""
    dialogue: list[dict] = []
    turns: list[int] = []
    user_idx = 0
    for m in messages:
        mtype = getattr(m, "type", None)
        content = getattr(m, "content", "")
        if isinstance(content, list):
            parts = [
                str(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = " ".join(parts)
        content = (content or "").strip()
        if mtype == "human" and content:
            user_idx += 1
            turns.append(user_idx)
            dialogue.append({"role": "user", "content": content})
        elif mtype == "ai" and content:
            dialogue.append({"role": "assistant", "content": content})
    return dialogue, turns


def _sanitize_slug(slug: str) -> str:
    """把 LLM 生成的 slug 清洗为合法 page_id（小写字母/数字/连字符）。"""
    s = re.sub(r"[^a-z0-9_-]+", "-", (slug or "").lower()).strip("-_")
    if s:
        return s[:128]
    import time
    return f"wiki-{int(time.time())}"


def _summarize(model, dialogue: list[dict], title_hint: str | None = None, config=None) -> WikiSummary | None:
    """用 LLM 把对话提炼为结构化 wiki 草稿，失败返回 None。"""
    conv = "\n".join(f"【{d['role']}】{d['content']}" for d in dialogue)
    hint = f"（期望标题参考：{title_hint}）" if title_hint else ""
    try:
        return models.invoke_structured(model, WikiSummary, [
            ("system", _SUMMARIZE_SYSTEM),
            ("human", f"请把以下学习对话总结为一篇体系化的 wiki 文章{hint}：\n\n{conv}"),
        ], config)
    except Exception:
        return None


async def summarize_thread_to_wiki(
    checkpointer, store, user_id: str, thread_id: str, model=None,
    title_hint: str | None = None, config=None,
) -> dict:
    """把某会话总结为 wiki 草稿，返回 ``{"ok": True, "page": meta}`` 或 ``{"ok": False, "reason": ...}``。"""
    if checkpointer is None or store is None or not user_id or not thread_id:
        return {"ok": False, "reason": "unavailable"}

    try:
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        state = await checkpointer.aget_state(config)
    except Exception:
        return {"ok": False, "reason": "read_failed"}

    messages = (state.values or {}).get("messages", []) if state else []
    dialogue, turns = _extract_dialogue(messages)
    if not turns:
        return {"ok": False, "reason": "empty"}

    summary = _summarize(model or models.get_planner_model(), dialogue, title_hint, config)
    if summary is None:
        return {"ok": False, "reason": "summarize_failed"}

    page = WikiPage(
        page_id=_sanitize_slug(summary.slug),
        title=summary.title,
        content_markdown=summary.markdown,
        tags=summary.tags,
        author=WikiAuthor.AI,
        source_thread_id=thread_id,
        source_turns=turns,
        evidence=summary.evidence,
    )
    try:
        meta = wiki.create_page(store, user_id, page)
    except ValueError:
        return {"ok": False, "reason": "invalid_slug"}
    if meta is None:
        return {"ok": False, "reason": "store_unavailable"}
    return {"ok": True, "page": meta}
