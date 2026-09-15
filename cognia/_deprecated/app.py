"""Cognia Chainlit UI（流式对话 + HITL）。

把 graph、memory、checkpointer 接上电，浏览器体验真实教学闭环。

架构（重构后）：graph 不再是固定流水线，而是「对话 Agent ⇄ 学习引擎」循环。
1. 任意用户输入（问候/目标/回答/提问/困惑/换话题）→ 对话 Agent 理解并决定如何回应；
   判定为学习目标时交给学习引擎建模，判定为「需要评估」时交给学习引擎诊断/验证。
2. Agent 产生面向用户的回复后，在 `await_user` interrupt 暂停，回复渲染给用户；
   用户下一条输入用 `Command(resume=...)` 续跑。
3. 每轮结束把 `proficiency_deltas` 写入长期 Store（跨会话熟练度）。
4. 认知状态只能由学习引擎（诊断 → 验证 → 状态机）变更，Agent 无直写权限。

关键接线：`_persist_deltas` 把 graph 产出在 state 里的 Delta 写入
`("proficiency", user_id)` 命名空间，使「graph → Store → 下次会话读回」链路闭合。

匿名 user_id（clarifications Q6）：前端 cognia.js 生成 UUID → localStorage 持久化 →
同步写 cookie，后端从 session.environ 读回。session_id（thread_id）与 user_id 分离。
"""

import asyncio
import json
import os
import pathlib
import uuid

import chainlit as cl
from dotenv import load_dotenv

from cognia import memory
from cognia.schemas import ProficiencyEntry

# .env 加载由应用入口负责（models.py 约定，任务⑦ 落地）
load_dotenv()

# ---- 登录认证（Chainlit 用户身份）----
# Chainlit 在匿名模式（无认证回调）下 session.user 恒为 None：
#   1) thread 无法关联 userId / userIdentifier，历史对话无法按人归属；
#   2) 刷新后 resume_thread 因 session.user 为空直接拒绝恢复历史对话。
# 因此引入最轻量的固定账号密码认证，登录一次后由 cookie 自动保持。
# 账号密码从 .env 读取（CHAINLIT_AUTH_USERNAME / CHAINLIT_AUTH_PASSWORD），
# 未配置时回退 admin/admin，仅用于本地教学环境。
CHAINLIT_AUTH_USERNAME = os.getenv("CHAINLIT_AUTH_USERNAME", "admin")
CHAINLIT_AUTH_PASSWORD = os.getenv("CHAINLIT_AUTH_PASSWORD", "admin")


@cl.password_auth_callback
async def password_auth_callback(username: str, password: str) -> cl.User | None:
    if (username, password) == (CHAINLIT_AUTH_USERNAME, CHAINLIT_AUTH_PASSWORD):
        return cl.User(identifier=username, metadata={"role": "admin"})
    return None


# ---- 对话持久化（Chainlit DataLayer → 本地 SQLite）----
# 默认 Chainlit 的 BaseDataLayer 只存内存、刷新即丢。这里接入 SQLAlchemyDataLayer，
# 把所有会话、消息、反馈持久化到 data/chainlit.db，便于事后查询与分析历史对话。
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CHAINLIT_DB_PATH = PROJECT_ROOT / "data" / "chainlit.db"

# Chainlit SQLAlchemyDataLayer 要求的表结构（SQLite 兼容版）。
# 字段名必须与 chainlit/data/sql_alchemy.py 中 SQL 语句使用的列名完全一致；
# 类型统一简化为 TEXT（SQLite 弱类型，JSON/数组均以 TEXT 存储）。
_CHAINLIT_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
        "id" TEXT PRIMARY KEY,
        "identifier" TEXT NOT NULL UNIQUE,
        "metadata" TEXT,
        "createdAt" TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS threads (
        "id" TEXT PRIMARY KEY,
        "createdAt" TEXT,
        "name" TEXT,
        "userId" TEXT,
        "userIdentifier" TEXT,
        "tags" TEXT,
        "metadata" TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS steps (
        "id" TEXT PRIMARY KEY,
        "name" TEXT,
        "type" TEXT,
        "threadId" TEXT,
        "parentId" TEXT,
        "disableFeedback" TEXT,
        "streaming" TEXT,
        "waitForAnswer" TEXT,
        "isError" TEXT,
        "metadata" TEXT,
        "tags" TEXT,
        "input" TEXT,
        "output" TEXT,
        "createdAt" TEXT,
        "start" TEXT,
        "end" TEXT,
        "generation" TEXT,
        "showInput" TEXT,
        "defaultOpen" TEXT,
        "autoCollapse" TEXT,
        "command" TEXT,
        "modes" TEXT,
        "icon" TEXT,
        "language" TEXT,
        "indent" INTEGER
    )""",
    """CREATE TABLE IF NOT EXISTS elements (
        "id" TEXT PRIMARY KEY,
        "threadId" TEXT,
        "type" TEXT,
        "url" TEXT,
        "chainlitKey" TEXT,
        "name" TEXT,
        "display" TEXT,
        "objectKey" TEXT,
        "size" TEXT,
        "page" INTEGER,
        "language" TEXT,
        "forId" TEXT,
        "mime" TEXT,
        "props" TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS feedbacks (
        "id" TEXT PRIMARY KEY,
        "forId" TEXT,
        "threadId" TEXT,
        "value" INTEGER,
        "comment" TEXT
    )""",
]


@cl.data_layer
def get_data_layer():
    """返回 Chainlit 对话持久化层（SQLite）。

    每次 Chainlit 需要数据层时调用（首条消息前）。先用同步 engine 幂等建表，
    再返回异步 SQLAlchemyDataLayer 供运行时读写。
    """
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
    from sqlalchemy import create_engine, text

    CHAINLIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sync_engine = create_engine(f"sqlite:///{CHAINLIT_DB_PATH}")
    with sync_engine.begin() as conn:
        for stmt in _CHAINLIT_SCHEMA:
            conn.execute(text(stmt))
    sync_engine.dispose()

    return SQLAlchemyDataLayer(
        conninfo=f"sqlite+aiosqlite:///{CHAINLIT_DB_PATH}",
        show_logger=True,
    )


def _build_config(thread_id: str, user_id: str) -> dict:
    """构建 LangGraph runtime config（thread_id + 匿名 user_id 注入）。

    thread_id 标识一次会话（Checkpointer 恢复），user_id 标识一个人（Store 隔离）。
    """
    return {"configurable": {"thread_id": thread_id, "user_id": user_id}}


def _resolve_user_id() -> str:
    """解析匿名 user_id（跨会话持久化，clarifications Q6）。

    优先从浏览器 cookie（`cognia_user_id`）读回——该 cookie 由前端 public/cognia.js
    写入（读 localStorage 的 cognia_user_id，没有则生成并同步写 localStorage + cookie）。
    后端通过 Chainlit 的 session.environ["HTTP_COOKIE"] 读回，实现 New Chat 之间
    保持同一 user_id；读不到时兜底生成新 UUID（首次访问）。
    """
    environ = cl.context.session.environ or {}
    cookie_header = environ.get("HTTP_COOKIE", "") or ""
    for part in cookie_header.split(";"):
        part = part.strip()
        if part.startswith("cognia_user_id="):
            value = part.split("=", 1)[1]
            if value:
                return value
    return uuid.uuid4().hex


async def _persist_deltas(store, user_id: str, state: dict) -> None:
    """把 state 里的 proficiency_deltas 写入长期 Store。

    幂等：append_proficiency_delta 用 `point_id:timestamp` 唯一 key，重复写入覆盖
    同 key，不会产生重复条目（因此可安全地在每轮末尾全量重写）。

    异步：生产 store 是 AsyncPostgresStore，主事件循环线程里必须用 `await aput`，
    同步 `store.put` 会因 @_check_loop 抛 InvalidStateError。
    """
    for delta in state.get("proficiency_deltas", []):
        entry = ProficiencyEntry.model_validate(delta)
        await memory.aappend_proficiency_delta(store, user_id, entry)


async def _remember_profile(store, user_id: str, user_text: str) -> None:
    """后台提取用户画像偏好并写入长期 Store（记忆功能，不阻塞主对话）。

    用低成本 planner 模型从用户本轮表达中提取稳定偏好（语言 / 沟通风格等），
    增量写入 `("profile", user_id)`。失败静默忽略——记忆是增强能力，不应
    因为画像提取失败而打断教学主流程。
    """
    if not user_id or not user_text:
        return
    try:
        from cognia import models

        profiler = models.get_planner_model()
        result = profiler.invoke([
            ("system", (
                "你是用户画像提取器。从用户的一句话中提取稳定偏好，输出 JSON 对象。"
                "可选字段：language（偏好的语言）、communication_style（沟通风格，如简洁/详细）。"
                "没有明显偏好时输出空对象 {}。只输出 JSON，不要任何解释。"
            )),
            ("human", f"用户说：{user_text}"),
        ])
        content = result.content if hasattr(result, "content") else str(result)
        data = models._extract_json(content)
        if isinstance(data, dict):
            for key, value in data.items():
                if value:
                    await memory.aput_profile(store, user_id, key, {"value": str(value)})
    except Exception as exc:  # 画像提取失败不影响主流程
        print(f"[Cognia] 画像提取失败（忽略）：{exc}")


# ---- 内存记忆层进程级单例（本地无 Postgres 时的 fallback）----
# 关键：InMemorySaver / InMemoryStore 必须是进程级单例。若每次 on_chat_start /
# on_chat_resume 都新建，则刷新后新 saver 里没有旧 checkpoint，会话恢复（本次修复
# 的核心目标）在本地会静默失效——checkpoint 只在同一个 saver 实例内可见。
_inmemory_checkpointer = None
_inmemory_store = None

async def _init_memory():
    """初始化记忆层；Postgres 不可用时降级到内存（本地开发体验）。

    生产配了 LANGGRAPH_DATABASE_URL 就走 Postgres（Checkpointer + Store 持久化）；
    本地没起 Postgres 时降级到 InMemory，保证「浏览器体验」不被数据库阻塞。

    get_checkpointer() 返回 AsyncPostgresSaver，必须在事件循环里 await 创建；
    get_store() 保持同步 PostgresStore（graph 同步节点在 executor 线程里调用
    同步 store.search，天然线程安全）。

    InMemory 分支同样走进程级单例（见上方 _inmemory_* 变量），否则「刷新恢复」
    在本地环境因 saver 重建而失效。
    """
    global _inmemory_checkpointer, _inmemory_store
    if os.getenv("LANGGRAPH_DATABASE_URL"):
        try:
            checkpointer = await memory.get_checkpointer()
            store = await memory.get_store()
            return checkpointer, store
        except Exception as exc:  # 连接失败 / 缺依赖等，降级保体验
            print(f"[Cognia] Postgres 不可用，降级到内存记忆层：{exc}")
    if _inmemory_checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        _inmemory_checkpointer = InMemorySaver()
    if _inmemory_store is None:
        from langgraph.store.memory import InMemoryStore
        _inmemory_store = InMemoryStore()
    return _inmemory_checkpointer, _inmemory_store


async def _init_session() -> None:
    """初始化运行时上下文（新对话与恢复对话共用）。

    每次 on_chat_start / on_chat_resume 都重新编译图（agent / tools 依赖本会话的
    user_id 与 store），但 checkpointer / store 是进程级单例——LangGraph 会按
    config 里的 thread_id 从同一个 checkpointer 实例读回历史 messages，因此
    「重编译 + 同 thread_id」能正确恢复对话。
    """
    checkpointer, store = await _init_memory()
    user_id = _resolve_user_id()

    from cognia import models
    from cognia.react_agent import build_react_graph
    from cognia.tools import build_cognia_tools

    teacher_model = models.get_conversation_agent_model()
    tools = build_cognia_tools(store=store, user_id=user_id, teacher=teacher_model)
    agent = teacher_model.bind_tools(list(tools.values()))
    tools_by_name = {t.name: t for t in tools.values()}
    graph = build_react_graph(agent, tools_by_name, checkpointer=checkpointer)

    cl.user_session.set("store", store)
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("graph", graph)


@cl.on_chat_start
async def on_chat_start():
    await _init_session()
    # 新会话：thread_id 即当前 session 的 thread id
    cl.user_session.set("thread_id", cl.context.session.thread_id)

    await cl.Message(
        content="你好！我是 Cognia，你的 AI 学习教练。\n请告诉我你想学什么（例如「Spring AOP」），也可以随便聊聊。"
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """恢复历史会话：重建运行时上下文，messages 由 checkpointer 按 thread_id
    自动读回，无需手动恢复。

    thread_id 的权威来源是 thread["id"]（Chainlit 回调传入的历史 thread id）。
    """
    await _init_session()
    cl.user_session.set("thread_id", thread["id"])


@cl.on_message
async def on_message(message: cl.Message):
    from langchain_core.messages import HumanMessage

    store = cl.user_session.get("store")
    user_id = cl.user_session.get("user_id")
    graph = cl.user_session.get("graph")

    thread_id = cl.context.session.thread_id
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

    # 后台记忆：从用户本轮表达中提取稳定偏好写 profile（不阻塞主流程）
    asyncio.create_task(_remember_profile(store, user_id, message.content))

    # 最终回答消息：边生成边流式（行业标准 SSE）
    reply_msg = cl.Message(content="")
    await reply_msg.send()

    # 当前思考块（每轮 LLM 生成一个独立 Step，默认展开、逐字可见）
    thinking_state: dict = {"step": None}
    # 工具卡片映射：call_id -> Step（并行同名工具也能正确关联）
    tool_steps: dict = {}

    async def _close_thinking() -> None:
        """结束当前思考块：update 后前端停止 loading，内容定格、可手动收起。"""
        if thinking_state["step"] is not None:
            await thinking_state["step"].update()
            thinking_state["step"] = None

    async def on_reasoning(text: str) -> None:
        # 新一轮思考开始：还没有思考块则创建（默认展开，逐字显示）
        if thinking_state["step"] is None:
            step = cl.Step(name="思考过程", type="run", default_open=True)
            await step.send()
            thinking_state["step"] = step
        await thinking_state["step"].stream_token(text)

    async def on_text(text: str) -> None:
        await reply_msg.stream_token(text)

    async def on_tool_start(call_id: str, name: str, args: dict) -> None:
        # 工具卡片：显示工具名 + 入参（show_input）+ 输出结果（默认展开）
        step = cl.Step(
            name=f"工具 · {name}",
            type="tool",
            show_input=True,
            default_open=True,
        )
        step.input = json.dumps(args, ensure_ascii=False, indent=2)
        await step.send()
        tool_steps[call_id] = step

    async def on_tool_result(call_id: str, name: str, result: str) -> None:
        step = tool_steps.get(call_id)
        if step is not None:
            step.output = result
            await step.update()

    async def _dispatch(event: str, payload: dict) -> None:
        """把图节点发来的 custom 事件分发到 UI 回调。"""
        if event == "reasoning":
            await on_reasoning(payload.get("text", ""))
        elif event == "text":
            await on_text(payload.get("text", ""))
        elif event == "tool_start":
            await on_tool_start(
                payload.get("call_id", ""),
                payload.get("name", ""),
                payload.get("args", {}),
            )
        elif event == "tool_result":
            await on_tool_result(
                payload.get("call_id", ""),
                payload.get("name", ""),
                payload.get("result", ""),
            )
        elif event == "turn_end":
            # 一轮 LLM 生成结束：收起当前思考块
            await _close_thinking()

    try:
        async for chunk in graph.astream(
            {"messages": [HumanMessage(content=message.content)]},
            config=config,
            stream_mode="custom",
        ):
            # custom 模式下 chunk 即节点 writer 发出的 {"event": ..., ...}
            if isinstance(chunk, dict) and "event" in chunk:
                await _dispatch(chunk["event"], chunk)
    except Exception as exc:  # LLM 调用失败等，给用户友好提示而非堆栈
        await reply_msg.stream_token(f"\n\n抱歉，处理时出错了：{exc}")
    finally:
        await _close_thinking()
        await reply_msg.update()
