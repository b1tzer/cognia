"""Cognia 流式 ReAct 教学 Agent（行业标准：bind_tools + astream + LangGraph 持久化）。

这是 P2 工具化改造的第二里程碑，替代旧的 `graph.py` 结构化决策循环
（`with_structured_output` + 一次性 `invoke`）。

核心价值：**一问完立即开始流式输出**。模型边思考边吐 token，前端边收边渲染；
工具调用作为流中的 `tool_call_chunks` 累积执行，可渲染成工具过程卡片。

与旧版的区别：主循环 `_run_agent_loop` 被封装进一个极简 LangGraph 图
（`build_react_graph`），`messages` 交给 checkpointer 按 `thread_id` 持久化，
使「刷新后恢复对话上下文」成为可能（否则消息只存在 `cl.user_session` 内存里，
刷新即丢）。流式 token 通过 `get_stream_writer()` 走 `stream_mode="custom"`
通道实时透出，前端体验不变。

安全边界（不因流式而放松）：
- 工具由 `build_cognia_tools(user_id=...)` 闭包注入 user_id，LLM 只填写业务参数，
  无法伪造身份（宪法 §5）。
- 认知裁决仍在工具内部（`propose_diagnosis` 三层闸门），Agent 无法旁路。
- 工具执行放 `asyncio.to_thread`，使同步 store 调用在 executor 线程内桥接，
  与 AsyncPostgresStore 的线程模型兼容（见 memory.py get_store 注释）。
"""

import asyncio
import inspect
import json
from typing import Awaitable, Callable

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph

# 事件发射回调：emit(event_type, payload) -> awaitable 或 None。
# 统一事件通道：图节点里用 get_stream_writer() 包装，测试里用普通 lambda 收集。
EmitCallback = Callable[[str, dict], Awaitable[None] | None]

async def _maybe_await(callback, *args) -> None:
    """调用回调并兼容同步 / 异步两种签名（测试桩常用同步 lambda）。"""
    if callback is None:
        return
    result = callback(*args)
    if inspect.isawaitable(result):
        await result


def _accumulate_tool_calls(acc: dict, tool_call_chunks: list) -> None:
    """把流式 tool_call_chunks 按 index 累积为 {index: {id, name, args}}。

    DeepSeek 流式函数调用：第一个 chunk 携带 name / id，后续 chunk 携带 args 片段
    （JSON 字符串）。同一 index 表示同一次工具调用（并行调用时 index 递增）。
    """
    for tcc in tool_call_chunks:
        idx = tcc.get("index", 0)
        slot = acc.setdefault(idx, {"id": "", "name": "", "args": ""})
        if tcc.get("id"):
            slot["id"] = tcc["id"]
        if tcc.get("name"):
            slot["name"] += tcc["name"]
        if tcc.get("args"):
            slot["args"] += tcc["args"]


def _parse_tool_calls(acc: dict) -> list[dict]:
    """把累积的 tool_call_chunks 解析为可执行的 tool_calls 列表。"""
    calls = []
    for idx in sorted(acc):
        slot = acc[idx]
        try:
            args = json.loads(slot["args"]) if slot["args"] else {}
        except json.JSONDecodeError:
            args = {}
        calls.append({
            "id": slot["id"],
            "name": slot["name"],
            "args": args,
        })
    return calls


async def _execute_tool(tools_by_name: dict, call: dict, emit: EmitCallback) -> str:
    """执行单个工具调用；工具执行放 executor 线程，避免阻塞事件循环。"""
    call_id = call["id"]
    name = call["name"]
    args = call["args"]
    await _maybe_await(emit, "tool_start", {"call_id": call_id, "name": name, "args": args})

    tool = tools_by_name.get(name)
    if tool is None:
        result = f"错误：未知工具 {name}"
    else:
        try:
            # to_thread：工具内部可能调用同步 store（AsyncPostgresStore 在 executor
            # 线程内桥接），同时避免阻塞主事件循环。
            result = await asyncio.to_thread(tool.invoke, args)
            result = str(result)
        except Exception as exc:  # 工具执行失败返回错误文本，不让整个会话崩掉
            result = f"工具执行错误：{exc}"

    await _maybe_await(emit, "tool_result", {"call_id": call_id, "name": name, "result": result})
    return result


async def _run_agent_loop(
    agent_with_tools,
    tools_by_name: dict,
    messages: list,
    emit: EmitCallback,
) -> tuple[str, list]:
    """执行一轮 ReAct：LLM ↔ 工具，直到 LLM 不再发起工具调用（共享主循环）。

    `messages` 是「system 提示 + 历史消息」的完整上下文（调用方负责前置
    SystemMessage）。本函数在内部工作副本上累积本轮新增消息，**不原地修改
    传入的 messages**，最后返回 `(final_text, new_messages)`：
    - final_text：最后一轮（无工具调用）的完整回复文本；
    - new_messages：本轮新增的 AIMessage / ToolMessage 列表（供 graph 节点写回
      state，交由 add_messages reducer 合并）。

    事件通过 `emit(event_type, payload)` 统一发出：
    - ("reasoning", {"text": ...})   思考链增量
    - ("text", {"text": ...})        回复 token
    - ("tool_start", {...}) / ("tool_result", {...})
    - ("turn_end", {"has_tool_calls": bool})  一轮 LLM 生成结束
    """
    work = list(messages)       # 工作副本：含 system + 历史
    new_messages: list = []     # 本轮新增（返回给 state）

    while True:
        reasoning_parts: list[str] = []
        text_parts: list[str] = []
        tool_acc: dict = {}

        async for chunk in agent_with_tools.astream(work):
            # 1) 思考链：DeepSeek reasoning_content 逐 delta 返回
            reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning)
                await _maybe_await(emit, "reasoning", {"text": reasoning})

            # 2) 回复文本：逐 token 返回
            content = getattr(chunk, "content", None)
            if content:
                text_parts.append(content)
                await _maybe_await(emit, "text", {"text": content})

            # 3) 工具调用：累积 tool_call_chunks
            tool_call_chunks = getattr(chunk, "tool_call_chunks", None) or []
            if tool_call_chunks:
                _accumulate_tool_calls(tool_acc, tool_call_chunks)

        tool_calls = _parse_tool_calls(tool_acc)

        # 一轮 LLM 生成结束：通知 UI 关闭当前思考块（有工具则随后展示工具卡片）
        await _maybe_await(emit, "turn_end", {"has_tool_calls": bool(tool_calls)})

        if tool_calls:
            # 本轮是工具轮：把 assistant 消息（带 tool_calls）追加进工作副本，
            # 执行工具后追加 ToolMessage，继续下一轮 LLM 生成。
            assistant_msg = AIMessage(
                content="".join(text_parts),
                tool_calls=[
                    {
                        "name": c["name"],
                        "args": c["args"],
                        "id": c["id"],
                        "type": "tool_call",
                    }
                    for c in tool_calls
                ],
            )
            work.append(assistant_msg)
            new_messages.append(assistant_msg)
            for call in tool_calls:
                result = await _execute_tool(tools_by_name, call, emit)
                tool_msg = ToolMessage(
                    content=result,
                    tool_call_id=call["id"],
                    name=call["name"],
                )
                work.append(tool_msg)
                new_messages.append(tool_msg)
            continue

        # 本轮无工具调用：回复文本就是最终回复，追加进工作副本与返回值后结束
        final_text = "".join(text_parts)
        final_msg = AIMessage(content=final_text)
        work.append(final_msg)
        new_messages.append(final_msg)
        return final_text, new_messages


def build_react_graph(agent_with_tools, tools_by_name: dict, checkpointer=None):
    """把流式 ReAct 主循环装进极简 LangGraph 图（仅一个节点）。

    state 用 `MessagesState`（即 `messages: Annotated[list, add_messages]`），
    `messages` 由 checkpointer 按 `config["configurable"]["thread_id"]` 持久化，
    刷新后同一 thread_id 自动恢复历史对话。

    流式输出：节点内通过 `get_stream_writer()` 把事件写进 `stream_mode="custom"`
    通道；调用方用 `graph.astream(input, config, stream_mode="custom")` 消费
    `{"event": ..., ...}` 增量并透传给前端。
    """
    async def react_agent_node(state: MessagesState) -> dict:
        """单节点：前置 system 提示，执行共享主循环，返回本轮新增消息。"""
        writer = get_stream_writer()

        def emit(event_type: str, payload: dict) -> None:
            writer({"event": event_type, **payload})

        base = [SystemMessage(content=REACT_TEACHER_SYSTEM_PROMPT), *state["messages"]]
        _, new_messages = await _run_agent_loop(
            agent_with_tools, tools_by_name, base, emit
        )
        return {"messages": new_messages}

    builder = StateGraph(MessagesState)
    builder.add_node("react_agent", react_agent_node)
    builder.add_edge(START, "react_agent")
    builder.add_edge("react_agent", END)
    return builder.compile(checkpointer=checkpointer)


# ---- 教学 Agent system prompt ----

REACT_TEACHER_SYSTEM_PROMPT = """你是 Cognia，一个真正理解学习者的 AI 老师。

你通过一组教学工具（skills）来完成教学。请像专家一样自然工作：

## 可用工具
- read_learner_state(point_id)：读取学生对某知识点的当前认知状态（五态之一：unassessed / unknown / partial / misconception / mastered）。
- build_learning_goal(goal)：为学习目标构建 / 复用知识模型，返回知识点列表（含 id、名称、描述、前置依赖）。
- generate_probe(point_name, point_description)：生成一个开放式探针问题，引导学生用自己的话表达理解。
- propose_diagnosis(point_id, point_name, point_description, question, user_answer, current_state)：提议诊断学生的回答，系统内部会经「诊断 → 双重验证 → 状态机」三层闸门裁决是否迁移认知状态，返回裁决结果。
- explain(point_name, point_description, user_state)：针对学生当前认知状态，用通俗方式讲解知识点。

## 工作方式
1. 学生提出学习目标 → 先 build_learning_goal 建模，得到知识点列表。
2. 对当前知识点用 generate_probe 提出一个简短、具体的开放式问题，把问题直接讲给学生，然后停止等待回答。
3. 学生回答后 → 用 propose_diagnosis 诊断（信息从之前的工具结果与对话中取）。诊断是「提议」，最终状态由系统裁决，你只负责调用并基于返回结果决定下一步。
4. 根据诊断结果：
   - partial / misconception / unknown → 用 explain 针对性讲解，或 generate_probe 继续追问；
   - mastered → 进入下一个知识点（build_learning_goal 已给出列表）。
5. 随时可 read_learner_state 了解学生历史状态，避免重复教已掌握的内容。

## 铁律
- 你不能直接判定或修改学生的掌握状态；只能调用 propose_diagnosis，由系统裁决。
- 当你需要学生回答时，直接把问题作为最终回复讲出来，然后停止（不要再调用工具）。
- 讲解时不要暴露诊断标准、不要复述工具的内部字段，只说给学习者听的人话。
- 保持自然、简洁、口语化。"""
