"""cognia.server 现役 ReAct 链路（LangGraph create_react_agent）集成测试。

验证现役接入层 `server.build_agent()` 组装的标准 ReAct agent：
1. 无工具调用时正常返回文本回复；
2. 有工具调用时 ToolNode 正确执行工具并进入下一轮；
3. checkpointer 按 thread_id 持久化，重编译后同 thread_id 仍能恢复历史。

全部使用继承 BaseChatModel 的 fake 模型离线运行，不依赖 DeepSeek API。
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr

from cognia import models
from cognia.server import build_agent


class _FakeModel(BaseChatModel):
    """按预设 AIMessage 序列依次返回的假模型。

    create_react_agent 会调用 `model.bind_tools(tools)` 再同步 `invoke`；
    这里 bind_tools 记录工具并返回 self，`_generate` 弹出预设脚本。
    """

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    _script: list = PrivateAttr(default_factory=list)
    _bound_tools: list | None = PrivateAttr(default=None)
    invocations: int = 0

    def __init__(self, script):
        super().__init__()
        self._script = list(script)

    def bind_tools(self, tools, **kwargs):
        self._bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invocations += 1
        msg = self._script.pop(0) if self._script else AIMessage(content="")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _install_fake_model(monkeypatch, script):
    model = _FakeModel(script)
    monkeypatch.setattr(models, "get_conversation_agent_model", lambda: model)
    return model


def test_plain_reply(monkeypatch):
    """无工具调用：直接返回文本回复。"""
    model = _install_fake_model(monkeypatch, [AIMessage(content="你好，我是 Cognia")])

    graph = build_agent()
    result = graph.invoke(
        {"messages": [HumanMessage(content="hi")]},
        {"configurable": {"thread_id": "t1"}},
    )

    assert result["messages"][-1].content == "你好，我是 Cognia"
    assert model.invocations == 1


def test_executes_tool(monkeypatch):
    """有工具调用：ToolNode 执行 read_learner_state，再进入最终回复轮。"""
    model = _install_fake_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_learner_state",
                        "args": {"point_id": "aop-concept"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="你目前是部分掌握"),
        ],
    )

    graph = build_agent()
    result = graph.invoke(
        {"messages": [HumanMessage(content="我 AOP 掌握得怎么样")]},
        {"configurable": {"thread_id": "t1"}},
    )

    messages = result["messages"]
    tool_msgs = [m for m in messages if m.type == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].name == "read_learner_state"
    assert "unassessed" in tool_msgs[0].content
    assert messages[-1].content == "你目前是部分掌握"
    assert model.invocations == 2


def test_checkpointer_recovers_history(monkeypatch):
    """同 saver + 同 thread_id：重编译 graph 后仍能恢复历史消息。"""
    model = _install_fake_model(
        monkeypatch,
        [AIMessage(content="第一次回复"), AIMessage(content="第二次回复")],
    )

    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t1"}}

    graph = build_agent(checkpointer=saver)
    graph.invoke({"messages": [HumanMessage(content="你好")]}, config)

    # 模拟前端刷新：重编译 graph，但复用同一 saver 实例与 thread_id
    graph2 = build_agent(checkpointer=saver)
    result = graph2.invoke({"messages": [HumanMessage(content="继续")]}, config)

    human_texts = [m.content for m in result["messages"] if m.type == "human"]
    assert human_texts == ["你好", "继续"]
    assert result["messages"][-1].content == "第二次回复"
