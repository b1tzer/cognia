"""会话管理路由（/threads CRUD + 历史回填 + 自动标题）。

权威数据在服务端 PostgreSQL：会话列表 = cognia_threads 元数据 ∪ checkpoints
真实会话；删除会话会同时清理 checkpointer 里的 checkpoint / blobs / writes。

与 AG-UI 接入层解耦：历史回填所需的 LangGraph graph 经 `app.state.graph`
注入（由 server.py 的 lifespan 设置），不依赖 server 的全局 agent 对象。
"""

import asyncio

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from ag_ui.core.types import RunAgentInput
from ag_ui_langgraph.utils import langchain_messages_to_agui

from cognia import models, threads

router = APIRouter()


class RenameThreadRequest(BaseModel):
    """重命名会话请求体。"""

    title: str


def _threads_available(request: Request) -> bool:
    """Postgres 降级为 InMemorySaver 时关闭会话管理，返回 False。"""
    return bool(getattr(request.app.state, "threads_available", False))


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


_TITLE_SYSTEM_PROMPT = (
    "你是会话标题生成助手。根据用户的第一条消息，提炼一个不超过15个字的"
    "精炼中文标题，直接概括用户的核心意图或问题。只输出标题本身，"
    "不要加引号、句号等标点，不要输出任何解释。"
)

def _fallback_title(text: str) -> str:
    """AI 生成失败时的降级标题：截断首条用户消息前 30 字。"""
    return text[:30] + ("…" if len(text) > 30 else "")

async def _generate_ai_title(text: str) -> str:
    """调用 LLM 生成精炼标题；失败或空结果抛异常，由调用方降级。"""
    model = models.get_planner_model()
    resp = await model.ainvoke([
        ("system", _TITLE_SYSTEM_PROMPT),
        ("human", text),
    ])
    title = (resp.content or "").strip()
    if not title:
        raise ValueError("模型返回空标题")
    return title[:30] + ("…" if len(title) > 30 else "")

async def _auto_title_task(pool, thread_id: str, text: str) -> None:
    """后台生成并写入会话标题（不阻塞对话流）。

    LLM 生成失败时降级为截断标题；写入失败只告警不中断，避免后台任务异常
    无人捕获（避免 "Task exception was never retrieved"）。
    """
    try:
        title = await _generate_ai_title(text)
    except Exception as exc:
        print(f"[Cognia] AI 生成会话标题失败，降级为截断：{exc}")
        title = _fallback_title(text)
    try:
        await threads.auto_title_if_empty(pool, thread_id, title)
    except Exception as exc:
        print(f"[Cognia] 写入自动标题失败：{exc}")

# 持有后台任务强引用，防止 asyncio.create_task 返回的任务被 GC 提前回收。
# 任务完成后通过 done_callback 自动从集合移除，不累积内存。
_background_tasks: set[asyncio.Task] = set()

def auto_title_thread(request: Request, input_data: RunAgentInput) -> None:
    """thread title 为空时，后台用 AI 精炼首条用户消息生成标题。

    异步方案：用 asyncio.create_task 后台执行，不阻塞当前对话请求的 SSE 流，
    本函数立即返回。auto_title_if_empty 在 SQL 层保证 title 非空时不覆盖，
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
    task = asyncio.create_task(_auto_title_task(app_state.pool, thread_id, text))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


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


@router.get("/threads")
async def list_threads_endpoint(request: Request):
    """列出当前用户全部会话（按最近活动时间降序）。"""
    if not _threads_available(request):
        return []
    return await threads.list_threads(request.app.state.pool)


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages_endpoint(thread_id: str, request: Request):
    """读取某会话的历史消息（权威数据在 PostgreSQL checkpointer）。

    供前端切换会话时回填聊天记录：CopilotKit 的 connect 走进程内存回放，
    对连接外部 LangGraph 后端的场景拿不到 checkpointer 历史，因此前端
    主动调用本端点读取并注入。返回 AG-UI 消息格式（user/assistant/tool）。
    """
    if not _threads_available(request):
        return {"messages": []}

    try:
        graph = request.app.state.graph
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


@router.post("/threads")
async def create_thread_endpoint(request: Request):
    """创建新会话，返回后端生成的 thread_id（元数据先落库）。"""
    if not _threads_available(request):
        raise HTTPException(status_code=503, detail="会话管理暂不可用（Postgres 未连接）")
    return await threads.create_thread(request.app.state.pool)


@router.patch("/threads/{thread_id}")
async def rename_thread_endpoint(
    thread_id: str, payload: RenameThreadRequest, request: Request
):
    """重命名会话（title 持久化到 cognia_threads 表）。"""
    if not _threads_available(request):
        raise HTTPException(status_code=503, detail="会话管理暂不可用（Postgres 未连接）")
    return await threads.rename_thread(request.app.state.pool, thread_id, payload.title)


@router.delete("/threads/{thread_id}")
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
