"""cognia.react_agent 流式 ReAct 循环单元测试。

使用假 async-agent（预设 chunk 序列）离线验证：
1. 思考 / 文本 token 逐块回调；
2. tool_call_chunks 正确累积为完整 tool_calls；
3. 工具执行 + ToolMessage 回填历史；
4. 最终回复返回与多轮循环终止。
"""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from cognia.react_agent import (
    _accumulate_tool_calls,
    _parse_tool_calls,
    stream_agent_turn,
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

def test_stream_agent_turn_two_rounds():
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

    reasoning_parts = []
    text_parts = []
    tool_starts = []
    tool_results = []
    turn_ends = []

    async def collect():
        final = await stream_agent_turn(
            agent,
            tools_by_name,
            messages,
            on_reasoning=lambda t: reasoning_parts.append(t),
            on_text=lambda t: text_parts.append(t),
            on_tool_start=lambda call_id, name, args: tool_starts.append((name, args)),
            on_tool_result=lambda call_id, name, result: tool_results.append((name, result)),
            on_llm_turn_end=lambda has_tool_calls: turn_ends.append(has_tool_calls),
        )
        return final

    final = asyncio.run(collect())

    assert final == "你目前对 AOP 是部分掌握"
    assert "".join(reasoning_parts) == "先查学生状态状态是 partial"
    assert "".join(text_parts) == "我查一下你的状态你目前对 AOP 是部分掌握"

    assert tool_starts == [("read_learner_state", {"point_id": "aop-concept"})]
    assert tool_results == [("read_learner_state", "partial")]

    # 轮次边界：第一轮（工具轮）True，第二轮（最终回答轮）False
    assert turn_ends == [True, False]

    # 历史回填：Human + AI(带 tool_calls) + Tool + AI(最终)
    assert len(messages) == 4
    assert isinstance(messages[1], AIMessage)
    assert messages[1].tool_calls[0]["name"] == "read_learner_state"
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].content == "partial"
    assert isinstance(messages[3], AIMessage)
    assert messages[3].content == "你目前对 AOP 是部分掌握"


def test_stream_agent_turn_unknown_tool_returns_error_text():
    """未知工具名：执行回调收到错误文本，会话不崩。"""
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

    async def collect():
        return await stream_agent_turn(
            agent,
            tools_by_name,
            messages,
            on_tool_result=lambda call_id, name, result: tool_results.append((name, result)),
        )

    final = asyncio.run(collect())
    assert final == "查完了"
    assert tool_results[0][0] == "no_such_tool"
    assert "未知工具" in tool_results[0][1]
