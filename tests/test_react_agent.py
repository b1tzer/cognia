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


def test_frontend_tool_interception(monkeypatch):
    """前端工具经 CopilotKitMiddleware 注入后，调用时被拦截转发而非后端执行。

    验证「三 / 五」（前端工具 + Generative UI）的后端关键闭环：
    1. 前端注册的工具被注入 LLM 的可用工具集（bind_tools 能看到它）；
    2. LLM 调用该前端工具时，不落 ToolNode 执行（无对应 tool 消息），
       而是被 middleware 拦截、记录到 copilotkit.intercepted_tool_calls，
       由接入层转成 AG-UI TOOL_CALL 事件交给前端渲染。
    """
    model = _install_fake_model(
        monkeypatch,
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "practice_choice",
                        "args": {"question": "Spring AOP 的核心是什么？"},
                        "id": "call_fe_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="请在卡片上作答"),
        ],
    )

    frontend_tool = {
        "name": "practice_choice",
        "description": "出一道交互式选择题，收集学生的答案",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
            },
        },
    }

    graph = build_agent()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="给我出个选择题")],
            "copilotkit": {"actions": [frontend_tool]},
        },
        {"configurable": {"thread_id": "t1"}},
    )

    # 1. 注入：bind_tools 收到的工具集合包含前端工具名。
    def _tool_name(t):
        if isinstance(t, dict):
            return t.get("name") or (t.get("function") or {}).get("name")
        return getattr(t, "name", None)

    bound_names = {_tool_name(t) for t in (model._bound_tools or [])}
    assert "practice_choice" in bound_names

    # 2. 拦截：不产生 practice_choice 的后端 tool 消息（不执行 ToolNode）。
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert all(m.name != "practice_choice" for m in tool_msgs)

    # 3. 前端工具调用被拦截后保留在 AIMessage.tool_calls（等待前端渲染并把
    #    结果回流），而不是被 ToolNode 消费掉。after_agent 会把 intercepted
    #    的 tool_call 恢复到原 assistant 消息上。
    ai_tool_calls = [
        tc.get("name")
        for m in result["messages"]
        if m.type == "ai"
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    assert "practice_choice" in ai_tool_calls
