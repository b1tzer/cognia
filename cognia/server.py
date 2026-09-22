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

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder
from copilotkit import CopilotKitMiddleware, CopilotKitState, LangGraphAGUIAgent

from cognia import memory, models, threads
from cognia.prompts.teacher import TEACHER_SYSTEM_PROMPT
from cognia.routers.knowledge_map import router as knowledge_map_router
from cognia.routers.threads import auto_title_thread, router as threads_router
from cognia.routers.wiki import router as wiki_router
from cognia.tools import build_cognia_tools


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
        checkpointer=checkpointer,  # 会话历史（summarize_session_to_wiki 读对话用）
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
    app.state.graph = _agent.graph
    yield


app = FastAPI(title="Cognia AG-UI Agent Server", lifespan=lifespan)


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
    await auto_title_thread(request, input_data)

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


# 挂载业务查询与会话管理 router（与 AG-UI 接入解耦）。
app.include_router(knowledge_map_router)
app.include_router(threads_router)
app.include_router(wiki_router)


def main() -> None:
    import uvicorn

    port = int(os.getenv("AGUI_PORT", "8123"))
    uvicorn.run("cognia.server:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()
