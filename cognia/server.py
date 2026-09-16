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
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder
from ag_ui_langgraph.utils import langchain_messages_to_agui
from copilotkit import CopilotKitMiddleware, CopilotKitState, LangGraphAGUIAgent

from cognia import memory, models, threads
from cognia.prompts.teacher import TEACHER_SYSTEM_PROMPT
from cognia.tools import build_cognia_tools


class RenameThreadRequest(BaseModel):
    """重命名会话请求体。"""

    title: str


def build_agent(checkpointer=None, store=None):
    """构建 Cognia 教学 ReAct agent（图）。

    - teacher 模型：对话 Agent（理解意图 + 决策 + 生成回复 + 工具调用）
    - tools：认知模型工具集（store 闭包注入，user_id 走 runtime context）
    - checkpointer：会话状态持久化。生产传 Postgres（跨刷新/重启恢复），
      不传时用 InMemorySaver（POC/测试）。
    - store：长期 Store（熟练度 / 知识模型持久化）。None 时跳过持久化
      （测试 / Postgres 降级），由 lifespan 显式告警，禁止静默丢弃。

    user_id 不在此注入：由每个请求在 endpoint 边界解析后，经
    `config["configurable"]["user_id"]`（runtime context）传给工具（宪法 §5）。
    """
    teacher = models.get_conversation_agent_model()
    tools = build_cognia_tools(
        store=store,         # 长期 Store；None 时工具内部跳过持久化
        teacher=teacher,
    )
    tool_list = list(tools.values())

    # 用 langchain.agents.create_agent（create_react_agent 已废弃）组装标准
    # ReAct agent，并挂上 CopilotKitMiddleware：它负责把前端通过 useFrontendTool
    # 注册的「前端工具」注入到 LLM 可用工具集，并在 LLM 调用前端工具时拦截、
    # 转发为 AG-UI 的 TOOL_CALL_* 事件给前端执行（前端渲染 Generative UI /
    # 执行 handler 后把结果回流）。state_schema 必须含 copilotkit 字段
    # （CopilotKitState），middleware 才能读回前端注册的工具清单。
    graph = create_agent(
        model=teacher,
        tools=tool_list,
        system_prompt=TEACHER_SYSTEM_PROMPT,
        middleware=[CopilotKitMiddleware()],
        state_schema=CopilotKitState,
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
        store = await memory.get_store()
        await threads.ensure_schema(pool)
        app.state.pool = pool
        app.state.checkpointer = checkpointer
        app.state.store = store
        app.state.threads_available = True
        print("[Cognia] 使用 Postgres 持久化会话状态与长期认知状态")
    except Exception as exc:  # Postgres 不可用 / 缺依赖等，降级保体验
        print(f"[Cognia] Postgres 不可用，降级到内存 checkpointer（长期认知状态不持久化）：{exc}")
        checkpointer = InMemorySaver()
        store = None
        app.state.checkpointer = checkpointer
        app.state.pool = None
        app.state.store = None
        app.state.threads_available = False
    _agent.graph = build_agent(checkpointer=checkpointer, store=store)
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

def _extract_user_id(input_data: RunAgentInput) -> str:
    """从 AG-UI 输入的 forwarded_props 解析匿名 user_id。

    前端经 forwarded_props 传匿名标识（MVP 无登录，clarifications Q6）。
    兼容 user_id / userId 两种 key（协议键名 snake_case，前端 JS 可能用 camelCase）。
    缺失时降级为 "local-user" 并显式告警，保持前端未升级前的兼容行为。
    """
    props = getattr(input_data, "forwarded_props", None)
    if not isinstance(props, dict):
        props = {}
    user_id = props.get("user_id") or props.get("userId")
    if not user_id:
        print("[Cognia] 未收到 user_id，降级为 local-user")
        return "local-user"
    return str(user_id)

# AG-UI 端点（path="/"）。与 ag_ui_langgraph.add_langgraph_fastapi_endpoint
# 等价，但多了「进入时自动生成会话标题」这一步；健康检查端点一并保留。
@app.post("/")
async def cognia_agent_endpoint(input_data: RunAgentInput, request: Request):
    await _auto_title_thread(request, input_data)

    accept_header = request.headers.get("accept")
    encoder = EventEncoder(accept=accept_header)

    # 每个请求构造独立 agent（复用 graph，注入逐请求 user_id），
    # 既避免并发请求共享 active_run 状态，又让 user_id 走 runtime context
    # （宪法 §5）。clone() 不接受 config 参数，故此处重建轻量包装对象。
    user_id = _extract_user_id(input_data)
    request_agent = LangGraphAGUIAgent(
        name=_agent.name,
        description=_agent.description,
        graph=_agent.graph,
        config={"configurable": {"user_id": user_id}},
    )

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


@app.get("/knowledge-map")
async def knowledge_map_endpoint(request: Request, user_id: str | None = None):
    """聚合当前匿名用户的知识版图（知识模型 + 熟练度），供认知图谱个人页渲染。

    - `user_id` 缺失 → 400（前端必须先建立匿名标识）。
    - store 为 None（Postgres 降级）→ 返回空版图 `goals: []`，不 500。
    - 每个 goal 聚合其全部知识点 + 每个知识点的当前熟练度（缺失补 unassessed）。
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="缺少 user_id 参数")

    store = getattr(request.app.state, "store", None)
    if store is None:
        return {"user_id": user_id, "goals": []}

    kms = await memory.alist_knowledge_models(store, user_id)
    proficiencies = await memory.alist_authoritative_proficiencies(store, user_id)

    goals = []
    for km in kms:
        points = km.get("points") or []
        goal_points = []
        goal_proficiencies = {}
        for p in points:
            pid = p.get("id")
            goal_points.append({
                "id": pid,
                "name": p.get("name"),
                "description": p.get("description"),
                "prerequisites": p.get("prerequisites") or [],
            })
            goal_proficiencies[pid] = proficiencies.get(pid, "unassessed")
        goals.append({
            "goal": km.get("goal", ""),
            "points": goal_points,
            "proficiencies": goal_proficiencies,
        })

    return {"user_id": user_id, "goals": goals}


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
