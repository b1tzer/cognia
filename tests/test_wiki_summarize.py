"""个人 Wiki 总结层（wiki_summarize）单元测试。

用 InMemoryStore（不依赖 Postgres）+ 临时 Git 目录（monkeypatch WIKI_REPO_ROOT）+
mock checkpointer / mock LLM（monkeypatch models.invoke_structured），验证：
1. 对话 → wiki 草稿（author=ai + 溯源字段 source_thread_id/evidence 落库）
2. 空对话返回「无可总结」
3. checkpointer / store / user_id 缺失安全降级
"""

import asyncio

import pytest
from langgraph.store.memory import InMemoryStore

from cognia import models, wiki
from cognia.schemas import WikiSummary
from cognia.wiki_summarize import summarize_thread_to_wiki


class _Msg:
    """mock 一条 checkpoint 消息（type + content，对齐 _extract_dialogue 读取的字段）。"""

    def __init__(self, mtype: str, content: str):
        self.type = mtype
        self.content = content


class _State:
    def __init__(self, messages):
        self.values = {"messages": messages}


class _FakeCheckpointer:
    """mock checkpointer：aget_state 返回预置的对话历史。"""

    def __init__(self, messages):
        self._messages = messages

    async def aget_state(self, config):
        return _State(self._messages)


@pytest.fixture
def repo_root(tmp_path, monkeypatch):
    root = tmp_path / "wiki"
    monkeypatch.setenv("WIKI_REPO_ROOT", str(root))
    return root


def _fake_summary() -> WikiSummary:
    return WikiSummary(
        title="Spring AOP 切面",
        slug="spring-aop-aspect",
        markdown="# AOP\n切面编程",
        evidence=["用户说：切面就是拦在方法前面的一段代码"],
        tags=["spring", "aop"],
    )


def test_summarize_produces_wiki(repo_root, monkeypatch):
    """正常：对话 → wiki 草稿，author=ai + 溯源字段正确落库。"""
    store = InMemoryStore()
    checkpointer = _FakeCheckpointer([
        _Msg("human", "切面就是拦在方法前面的一段代码，对吗？"),
        _Msg("ai", "对，切面（Aspect）是在不修改原代码的情况下横切关注点的模块。"),
        _Msg("human", "那 AOP 和 OOP 有什么区别？"),
        _Msg("ai", "OOP 纵向组织类，AOP 横向抽取公共关注点。"),
    ])
    monkeypatch.setattr(models, "invoke_structured", lambda *a, **k: _fake_summary())

    result = asyncio.run(summarize_thread_to_wiki(
        checkpointer, store, "u1", "t1", model=object(), title_hint="AOP",
    ))

    assert result["ok"] is True
    page = result["page"]
    assert page["author"] == "ai"
    assert page["source_thread_id"] == "t1"
    assert page["source_turns"] == [1, 2]
    assert page["evidence"] == ["用户说：切面就是拦在方法前面的一段代码"]

    # 正文已写 Git + 元数据可读回（author=ai）
    got = wiki.get_page(store, "u1", "spring-aop-aspect")
    assert got["content_markdown"] == "# AOP\n切面编程"
    assert got["author"] == "ai"


def test_summarize_empty_dialogue(repo_root):
    """边界：无用户实质内容 → 返回无可总结，不产出空页面。"""
    store = InMemoryStore()
    checkpointer = _FakeCheckpointer([])
    result = asyncio.run(summarize_thread_to_wiki(
        checkpointer, store, "u1", "t1", model=object(),
    ))
    assert result["ok"] is False
    assert result["reason"] == "empty"
    assert wiki.list_pages(store, "u1") == []


def test_summarize_unavailable(repo_root):
    """异常：checkpointer / store / user_id 缺失 → 安全降级，不抛异常。"""
    checkpointer = _FakeCheckpointer([_Msg("human", "hi")])
    assert asyncio.run(summarize_thread_to_wiki(
        None, InMemoryStore(), "u1", "t1",
    )) == {"ok": False, "reason": "unavailable"}
    assert asyncio.run(summarize_thread_to_wiki(
        checkpointer, None, "u1", "t1",
    )) == {"ok": False, "reason": "unavailable"}
    assert asyncio.run(summarize_thread_to_wiki(
        checkpointer, InMemoryStore(), None, "t1",
    )) == {"ok": False, "reason": "unavailable"}
