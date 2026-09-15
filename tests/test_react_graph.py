"""cognia.react_agent 极简图 + checkpointer 持久化测试。

核心验证「刷新后恢复对话上下文」的两条路径：
1. 同一个 graph 实例、同一个 thread_id 二次调用能读回历史 messages；
2. 模拟 on_chat_resume：新建 graph（重编译）但复用同一个 saver 实例、同一个
   thread_id，仍能读回历史——这正是 app.py 每次 _init_session 重编译图的场景，
   若 InMemorySaver 不是进程级单例，此用例会把「本地刷新修复无效」的问题逼出来。
"""

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from cognia.react_agent import build_react_graph


class _Chunk:
    """模拟 AIMessageChunk：仅提供主循环需要的 content / additional_kwargs / tool_call_chunks。"""

    def __init__(self, content="", reasoning="", tool_call_chunks=None):
        self.content = content
        self.additional_kwargs = {"reasoning_content": reasoning} if reasoning else {}
        self.tool_call_chunks = tool_call_chunks or []


class _StaticAgent:
    """每次 astream 都返回一条固定文本回复（无工具调用）。"""

    def __init__(self, reply="收到"):
        self.reply = reply

    async def astream(self, messages):
        yield _Chunk(content=self.reply)


def _human_texts(messages) -> list[str]:
    return [m.content for m in messages if getattr(m, "type", None) == "human"]


def test_graph_persists_messages_same_thread():
    """同一 graph + 同一 thread_id：第二次调用能读到第一次的 messages。"""
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t1"}}
    agent = _StaticAgent(reply="收到")

    async def run():
        graph = build_react_graph(agent, {}, checkpointer=saver)

        chunks = []
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content="你好")]},
            config=config,
            stream_mode="custom",
        ):
            chunks.append(chunk)

        async for chunk in graph.astream(
            {"messages": [HumanMessage(content="还记得我吗")]},
            config=config,
            stream_mode="custom",
        ):
            chunks.append(chunk)

        messages = graph.get_state(config).values["messages"]
        assert _human_texts(messages) == ["你好", "还记得我吗"]

        # 流式 text 事件确实通过 custom 通道透出
        text_events = [
            c["text"]
            for c in chunks
            if isinstance(c, dict) and c.get("event") == "text"
        ]
        assert text_events, "应至少有一条 text 流式事件"

    asyncio.run(run())


def test_graph_recovers_history_after_recompile_same_saver():
    """模拟 on_chat_resume：新 graph（重编译）+ 同 saver + 同 thread_id 仍能恢复。"""
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t1"}}

    async def run():
        # 第一次会话
        graph1 = build_react_graph(_StaticAgent(reply="第一次回复"), {}, checkpointer=saver)
        async for _ in graph1.astream(
            {"messages": [HumanMessage(content="你好")]},
            config=config,
            stream_mode="custom",
        ):
            pass

        # 模拟 on_chat_resume：重编译 graph，但复用同一个 saver 实例与 thread_id
        graph2 = build_react_graph(_StaticAgent(reply="第二次回复"), {}, checkpointer=saver)
        async for _ in graph2.astream(
            {"messages": [HumanMessage(content="继续")]},
            config=config,
            stream_mode="custom",
        ):
            pass

        messages = graph2.get_state(config).values["messages"]
        assert _human_texts(messages) == ["你好", "继续"]

    asyncio.run(run())
