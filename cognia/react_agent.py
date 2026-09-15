"""Cognia 流式 ReAct 教学 Agent（行业标准：bind_tools + astream）。

这是 P2 工具化改造的第二里程碑，替代旧的 `graph.py` 结构化决策循环
（`with_structured_output` + 一次性 `invoke`）。

核心价值：**一问完立即开始流式输出**。模型边思考边吐 token，前端边收边渲染；
工具调用作为流中的 `tool_call_chunks` 累积执行，可渲染成工具过程卡片。

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

from langchain_core.messages import AIMessage, ToolMessage

# 回调类型：都是 async 可等待函数
ReasoningCallback = Callable[[str], Awaitable[None]]
TextCallback = Callable[[str], Awaitable[None]]
ToolStartCallback = Callable[[str, str, dict], Awaitable[None]]  # (call_id, name, args)
ToolResultCallback = Callable[[str, str, str], Awaitable[None]]  # (call_id, name, result)
TurnEndCallback = Callable[[bool], Awaitable[None]]  # has_tool_calls：一轮 LLM 生成结束


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


async def _execute_tool(tools_by_name: dict, call: dict, on_start, on_result) -> str:
    """执行单个工具调用；工具执行放 executor 线程，避免阻塞事件循环。"""
    call_id = call["id"]
    name = call["name"]
    args = call["args"]
    await _maybe_await(on_start, call_id, name, args)

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

    await _maybe_await(on_result, call_id, name, result)
    return result


async def stream_agent_turn(
    agent_with_tools,
    tools_by_name: dict,
    messages: list,
    *,
    on_reasoning: ReasoningCallback | None = None,
    on_text: TextCallback | None = None,
    on_tool_start: ToolStartCallback | None = None,
    on_tool_result: ToolResultCallback | None = None,
    on_llm_turn_end: TurnEndCallback | None = None,
) -> str:
    """执行一轮 ReAct：LLM ↔ 工具，直到 LLM 不再发起工具调用。

    流式消费 `agent_with_tools.astream(messages)`，把三类增量实时回调出去：
    - reasoning_content（思考链）→ on_reasoning
    - content（回复文本）→ on_text
    - tool_call_chunks（工具调用）→ 累积后执行，触发 on_tool_start / on_tool_result
    - 每轮 LLM 生成结束 → on_llm_turn_end(has_tool_calls)，供 UI 区分「思考 / 工具 /
      最终回答」的轮次边界（行业标准 Agent 的节奏感）

    返回：最后一轮（无工具调用）的完整回复文本；纯工具轮返回空串。
    """
    while True:
        reasoning_parts: list[str] = []
        text_parts: list[str] = []
        tool_acc: dict = {}

        async for chunk in agent_with_tools.astream(messages):
            # 1) 思考链：DeepSeek reasoning_content 逐 delta 返回
            reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")
            if reasoning:
                reasoning_parts.append(reasoning)
                await _maybe_await(on_reasoning, reasoning)

            # 2) 回复文本：逐 token 返回
            content = getattr(chunk, "content", None)
            if content:
                text_parts.append(content)
                await _maybe_await(on_text, content)

            # 3) 工具调用：累积 tool_call_chunks
            tool_call_chunks = getattr(chunk, "tool_call_chunks", None) or []
            if tool_call_chunks:
                _accumulate_tool_calls(tool_acc, tool_call_chunks)

        tool_calls = _parse_tool_calls(tool_acc)

        # 一轮 LLM 生成结束：通知 UI 关闭当前思考块（有工具则随后展示工具卡片）
        await _maybe_await(on_llm_turn_end, bool(tool_calls))

        if tool_calls:
            # 本轮是工具轮：把 assistant 消息（带 tool_calls）追加进历史，
            # 执行工具后追加 ToolMessage，继续下一轮 LLM 生成。
            messages.append(AIMessage(
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
            ))
            for call in tool_calls:
                result = await _execute_tool(
                    tools_by_name, call, on_tool_start, on_tool_result
                )
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=call["id"],
                    name=call["name"],
                ))
            continue

        # 本轮无工具调用：回复文本就是最终回复，追加进历史后返回
        final_text = "".join(text_parts)
        messages.append(AIMessage(content=final_text))
        return final_text


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
