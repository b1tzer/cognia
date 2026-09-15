"""Cognia AG-UI 后端服务（CopilotKit 集成）。

用 LangGraph 官方 `create_react_agent`（prebuilt ReAct）把教学工具组装成标准
ReAct agent，再用 CopilotKit 的 `LangGraphAGUIAgent` 包装成 AG-UI 协议端点，
供 Next.js + CopilotKit 前端消费。

架构：
    Next.js 前端 (CopilotKit React)
        ↕ AG-UI (SSE, /api/copilotkit)
    CopilotKit Runtime (route.ts, LangGraphHttpAgent)
        ↕ HTTP/SSE
    本服务 (FastAPI + LangGraphAGUIAgent)
        ↕
    create_react_agent(graph) + build_cognia_tools()

关键点：
- 思考链（DeepSeek reasoning_content）由 ag-ui-langgraph 映射为 AG-UI 的
  REASONING_* 事件，前端流式显示思考过程。
- 工具调用由 ToolNode 映射为 TOOL_CALL_* 事件，前端自动渲染工具卡片
  （入参 → 执行中 → 结果）。
- 认知裁决权仍在工具内部（propose_diagnosis 三层闸门），不因流式而放松。

启动：
    uv run python -m cognia.server
    （监听 0.0.0.0:8123，AG-UI 端点路径 /）
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder
from ag_ui_langgraph.utils import langchain_messages_to_agui
from copilotkit import LangGraphAGUIAgent

from cognia import memory, models, threads
from cognia.react_agent import REACT_TEACHER_SYSTEM_PROMPT
from cognia.tools import build_cognia_tools


class RenameThreadRequest(BaseModel):
    """重命名会话请求体。"""

    title: str


def build_agent(checkpointer=None, user_id: str = "local-user"):
    """构建 Cognia 教学 ReAct agent（图）。

    - teacher 模型：对话 Agent（理解意图 + 决策 + 生成回复 + 工具调用）
    - tools：认知模型工具集（user_id 闭包注入，LLM 不可伪造身份）
    - checkpointer：会话状态持久化。生产传 Postgres（跨刷新/重启恢复），
      不传时用 InMemorySaver（POC/测试）。

    注意：user_id 目前为固定值（POC 阶段）。多用户身份注入需在请求边界
    解析并逐请求构建 agent（见 LangGraphAGUIAgent config 参数），属后续安全加固项。
    """
    teacher = models.get_conversation_agent_model()
    tools = build_cognia_tools(
        store=None,          # POC：不接长期 Store，验证事件流
        user_id=user_id,     # 身份闭包注入，LLM 无法伪造
        teacher=teacher,
    )
    tool_list = list(tools.values())

    graph = create_react_agent(
        model=teacher,
        tools=tool_list,
        prompt=REACT_TEACHER_SYSTEM_PROMPT,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
    return graph


# 占位 agent：endpoint 在模块加载时就要绑定 agent。这里先用 InMemorySaver 构建
# 一个占位 graph，lifespan 启动时再用持久化 checkpointer（Postgres）重建并原地
# 替换 _agent.graph（endpoint 闭包捕获的是 _agent 对象，clone 时读它的 graph）。
_agent = LangGraphAGUIAgent(
    name="cognia_teacher",
    description="Cognia AI 主动学习教练：诊断认知盲区并动态引导掌握知识点",
    graph=build_agent(),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化持久化 checkpointer，替换占位 graph。

    对话历史跨「浏览器刷新 / 服务 reload / 服务重启」持久化的关键：
    checkpointer 必须是 Postgres（进程外）。InMemorySaver 只活在进程内存里，
    --reload 或重启即全丢。

    同时把 checkpointer 与连接池挂到 app.state，供 /threads 会话管理端点复用；
    Postgres 不可用时降级为内存 checkpointer，并关闭会话管理能力。
    """
    try:
        pool = await memory.get_pool()
        checkpointer = await memory.get_checkpointer()
        await threads.ensure_schema(pool)
        app.state.pool = pool
        app.state.checkpointer = checkpointer
        app.state.threads_available = True
        print("[Cognia] 使用 Postgres checkpointer 持久化会话状态")
    except Exception as exc:  # Postgres 不可用 / 缺依赖等，降级保体验
        print(f"[Cognia] Postgres 不可用，降级到内存 checkpointer：{exc}")
        checkpointer = InMemorySaver()
        app.state.checkpointer = checkpointer
        app.state.pool = None
        app.state.threads_available = False
    _agent.graph = build_agent(checkpointer=checkpointer)
    yield


app = FastAPI(title="Cognia AG-UI Agent Server", lifespan=lifespan)


def _extract_first_user_text(input_data: RunAgentInput) -> str | None:
    """从 AG-UI 输入中提取首条用户文本消息，用于生成会话标题。"""
    for msg in input_data.messages or []:
        role = getattr(msg, "role", None)
        if role != "user":
            continue
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        return text
    return None


async def _auto_title_thread(request: Request, input_data: RunAgentInput) -> None:
    """thread title 为空时，用首条用户消息截断作为临时标题。

    只调用一次即可：auto_title_if_empty 在 SQL 层保证 title 非空时不覆盖，
    用户手动重命名后保持不被改写。
    """
    app_state = request.app.state
    if not getattr(app_state, "threads_available", False):
        return
    thread_id = getattr(input_data, "thread_id", None)
    if not thread_id:
        return
    text = _extract_first_user_text(input_data)
    if not text:
        return
    title = text[:30] + ("…" if len(text) > 30 else "")
    await threads.auto_title_if_empty(app_state.pool, thread_id, title)


# AG-UI 端点（path="/"）。与 ag_ui_langgraph.add_langgraph_fastapi_endpoint
# 等价，但多了「进入时自动生成会话标题」这一步；健康检查端点一并保留。
@app.post("/")
async def cognia_agent_endpoint(input_data: RunAgentInput, request: Request):
    await _auto_title_thread(request, input_data)

    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)

    # 每个请求 clone 独立 agent，避免并发请求共享 active_run 状态。
    request_agent = _agent.clone()

    async def event_generator():
        async for event in request_agent.run(input_data):
            yield encoder.encode(event)

    return StreamingResponse(
        event_generator(),
        media_type=encoder.get_content_type(),
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent": {"name": _agent.name},
    }


# ---- 会话管理端点（多会话列表 / 新建 / 重命名 / 删除）----
# 权威数据在服务端 PostgreSQL：会话列表 = cognia_threads 元数据 ∪ checkpoints
# 真实会话；删除会话会同时清理 checkpointer 里的 checkpoint / blobs / writes。

def _threads_available(request: Request) -> bool:
    """Postgres 降级为 InMemorySaver 时关闭会话管理，返回 False。"""
    return bool(getattr(request.app.state, "threads_available", False))


def _history_to_agui(messages) -> list:
    """把 checkpoint 历史消息转换为 AG-UI 消息，并补出思考过程。

    ag-ui-langgraph 的 ``langchain_messages_to_agui`` 只处理 AIMessage 的
    ``content`` 列表里的 reasoning block；而 DeepSeek 的思考链存放在
    ``additional_kwargs.reasoning_content``，不在这里补出来的话，刷新页面
    回填历史时就看不到思考过程。这里在每个 AIMessage 之前插入一条
    role="reasoning" 的 dict（与前端 AG-UI 消息结构一致）。
    """
    out = []
    for raw in messages:
        if isinstance(raw, AIMessage):
            reasoning = (raw.additional_kwargs or {}).get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                out.append({
                    "id": f"{raw.id}-reasoning",
                    "role": "reasoning",
                    "content": reasoning,
                })
        out.extend(langchain_messages_to_agui([raw]))
    return out


@app.get("/threads")
async def list_threads_endpoint(request: Request):
    """列出当前用户全部会话（按最近活动时间降序）。"""
    if not _threads_available(request):
        return []
    return await threads.list_threads(request.app.state.pool)


@app.get("/threads/{thread_id}/messages")
async def get_thread_messages_endpoint(thread_id: str, request: Request):
    """读取某会话的历史消息（权威数据在 PostgreSQL checkpointer）。

    供前端切换会话时回填聊天记录：CopilotKit 的 connect 走进程内存回放，
    对连接外部 LangGraph 后端的场景拿不到 checkpointer 历史，因此前端
    主动调用本端点读取并注入。返回 AG-UI 消息格式（user/assistant/tool）。
    """
    if not _threads_available(request):
        return {"messages": []}

    try:
        graph = _agent.graph
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        state = await graph.aget_state(config)
        messages = (state.values or {}).get("messages", [])
        agui_messages = _history_to_agui(messages)
    except Exception as exc:
        # 单个会话 checkpoint 数据异常 / 转换失败不应拖垮整个端点。
        print(f"[Cognia] 读取会话 {thread_id} 历史失败：{exc}")
        return {"messages": []}

    # 过滤掉 system 消息（系统提示词无需回显），并以 camelCase alias 序列化
    # （AG-UI 前端消息字段为 toolCalls / toolCallId 等 camelCase）。
    # reasoning 消息是我们手工构造的 dict，没有 model_dump，直接透传。
    result = []
    for m in agui_messages:
        if getattr(m, "role", None) == "system":
            continue
        result.append(m.model_dump(by_alias=True, mode="json") if hasattr(m, "model_dump") else m)
    return {"messages": result}


@app.post("/threads")
async def create_thread_endpoint(request: Request):
    """创建新会话，返回后端生成的 thread_id（元数据先落库）。"""
    if not _threads_available(request):
        raise HTTPException(status_code=503, detail="会话管理暂不可用（Postgres 未连接）")
    return await threads.create_thread(request.app.state.pool)


@app.patch("/threads/{thread_id}")
async def rename_thread_endpoint(
    thread_id: str, payload: RenameThreadRequest, request: Request
):
    """重命名会话（title 持久化到 cognia_threads 表）。"""
    if not _threads_available(request):
        raise HTTPException(status_code=503, detail="会话管理暂不可用（Postgres 未连接）")
    return await threads.rename_thread(request.app.state.pool, thread_id, payload.title)


@app.delete("/threads/{thread_id}")
async def delete_thread_endpoint(thread_id: str, request: Request):
    """删除会话及其 checkpoint（含 blobs / writes）。"""
    if not _threads_available(request):
        raise HTTPException(status_code=503, detail="会话管理暂不可用（Postgres 未连接）")
    await threads.delete_thread(
        request.app.state.pool,
        request.app.state.checkpointer,
        thread_id,
    )
    return {"deleted": True, "thread_id": thread_id}


def main() -> None:
    import uvicorn

    port = int(os.getenv("AGUI_PORT", "8123"))
    uvicorn.run("cognia.server:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()
