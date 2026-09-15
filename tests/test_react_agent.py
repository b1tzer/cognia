"""cognia.react_agent 共享主循环单元测试。

使用假 async-agent（预设 chunk 序列）离线验证：
1. tool_call_chunks 正确累积为完整 tool_calls；
2. 思考 / 文本 token 通过统一 emit 事件通道逐块发出；
3. 工具执行 + 本轮新增消息返回（不原地修改传入 history）；
4. 最终回复返回与多轮循环终止。
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from cognia.react_agent import (
    _accumulate_tool_calls,
    _parse_tool_calls,
    _run_agent_loop,
)


class _Chunk:
    """模拟 AIMessageChunk：content / additional_kwargs.reasoning_content / tool_call_chunks。"""

    def __init__(self, content="", reasoning="", tool_call_chunks=None):
        self.content = content
        self.additional_kwargs = {"reasoning_content": reasoning} if reasoning else {}
        self.tool_call_chunks = tool_call_chunks or []


class _FakeAgent:
    """按预设 chunk 轮次返回的假 bind_tools 模型。"""

    def __init__(self, rounds):
        self._rounds = list(rounds)
        self.messages_seen = []

    async def astream(self, messages):
        self.messages_seen.append(messages)
        assert self._rounds, "fake agent rounds exhausted"
        for chunk in self._rounds.pop(0):
            yield chunk


@tool
def read_learner_state(point_id: str) -> str:
    """读取当前用户对某知识点的认知状态。"""
    return "partial"


# ---- 纯函数：tool_call_chunks 累积与解析 ----

def test_accumulate_and_parse_tool_calls():
    acc = {}
    chunks = [
        {"index": 0, "name": "read_learner_state", "args": "", "id": "call_1", "type": "tool_call_chunk"},
        {"index": 0, "name": None, "args": '{"point_id": ', "id": None, "type": "tool_call_chunk"},
        {"index": 0, "name": None, "args": '"aop-concept"}', "id": None, "type": "tool_call_chunk"},
    ]
    _accumulate_tool_calls(acc, chunks)
    calls = _parse_tool_calls(acc)

    assert len(calls) == 1
    assert calls[0]["name"] == "read_learner_state"
    assert calls[0]["id"] == "call_1"
    assert calls[0]["args"] == {"point_id": "aop-concept"}


def test_parse_tool_calls_bad_json_tolerated():
    """args 非法 JSON 时降级为空 dict，不抛异常。"""
    acc = {0: {"id": "call_1", "name": "x", "args": "{bad json"}}
    calls = _parse_tool_calls(acc)
    assert calls == [{"id": "call_1", "name": "x", "args": {}}]


# ---- 集成：两轮 ReAct（工具轮 + 最终回复轮）----

def test_run_agent_loop_two_rounds():
    rounds = [
        # 第一轮：思考 → 文本 → 工具调用
        [
            _Chunk(reasoning="先查学生状态"),
            _Chunk(content="我查一下你的状态"),
            _Chunk(tool_call_chunks=[
                {"index": 0, "name": "read_learner_state", "args": "", "id": "call_1", "type": "tool_call_chunk"},
            ]),
            _Chunk(tool_call_chunks=[
                {"index": 0, "name": None, "args": '{"point_id": "aop-concept"}', "id": None, "type": "tool_call_chunk"},
            ]),
        ],
        # 第二轮：思考 → 最终回复（无工具）
        [
            _Chunk(reasoning="状态是 partial"),
            _Chunk(content="你目前对 AOP 是部分掌握"),
        ],
    ]
    agent = _FakeAgent(rounds)
    tools_by_name = {"read_learner_state": read_learner_state}
    messages = [HumanMessage(content="我 AOP 掌握得怎么样？")]

    events = []

    def emit(event_type, payload):
        events.append((event_type, payload))

    async def collect():
        final, new_messages = await _run_agent_loop(agent, tools_by_name, messages, emit)
        return final, new_messages

    final, new_messages = asyncio.run(collect())

    reasoning_parts = []
    text_parts = []
    tool_starts = []
    tool_results = []
    turn_ends = []
    for event_type, payload in events:
        if event_type == "reasoning":
            reasoning_parts.append(payload["text"])
        elif event_type == "text":
            text_parts.append(payload["text"])
        elif event_type == "tool_start":
            tool_starts.append((payload["name"], payload["args"]))
        elif event_type == "tool_result":
            tool_results.append((payload["name"], payload["result"]))
        elif event_type == "turn_end":
            turn_ends.append(payload["has_tool_calls"])

    assert final == "你目前对 AOP 是部分掌握"
    assert "".join(reasoning_parts) == "先查学生状态状态是 partial"
    assert "".join(text_parts) == "我查一下你的状态你目前对 AOP 是部分掌握"

    assert tool_starts == [("read_learner_state", {"point_id": "aop-concept"})]
    assert tool_results == [("read_learner_state", "partial")]

    # 轮次边界：第一轮（工具轮）True，第二轮（最终回答轮）False
    assert turn_ends == [True, False]

    # 本轮新增消息：AI(带 tool_calls) + Tool + AI(最终)
    assert len(new_messages) == 3
    assert isinstance(new_messages[0], AIMessage)
    assert new_messages[0].tool_calls[0]["name"] == "read_learner_state"
    assert isinstance(new_messages[1], ToolMessage)
    assert new_messages[1].content == "partial"
    assert isinstance(new_messages[2], AIMessage)
    assert new_messages[2].content == "你目前对 AOP 是部分掌握"

    # 传入 history 不被原地修改（共享主循环的工作副本语义）
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)


def test_run_agent_loop_unknown_tool_returns_error_text():
    """未知工具名：执行事件收到错误文本，会话不崩。"""
    rounds = [
        [
            _Chunk(tool_call_chunks=[
                {"index": 0, "name": "no_such_tool", "args": '{}', "id": "call_x", "type": "tool_call_chunk"},
            ]),
        ],
        [
            _Chunk(content="查完了"),
        ],
    ]
    agent = _FakeAgent(rounds)
    tools_by_name = {}
    messages = [HumanMessage(content="hi")]

    tool_results = []

    def emit(event_type, payload):
        if event_type == "tool_result":
            tool_results.append((payload["name"], payload["result"]))

    async def collect():
        final, _ = await _run_agent_loop(agent, tools_by_name, messages, emit)
        return final

    final = asyncio.run(collect())
    assert final == "查完了"
    assert tool_results[0][0] == "no_such_tool"
    assert "未知工具" in tool_results[0][1]
