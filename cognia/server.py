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

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent

from cognia import models
from cognia.react_agent import REACT_TEACHER_SYSTEM_PROMPT
from cognia.tools import build_cognia_tools

app = FastAPI(title="Cognia AG-UI Agent Server")


def build_agent(user_id: str = "local-user"):
    """构建 Cognia 教学 ReAct agent（图）。

    - teacher 模型：对话 Agent（理解意图 + 决策 + 生成回复 + 工具调用）
    - tools：认知模型工具集（user_id 闭包注入，LLM 不可伪造身份）
    - checkpointer：会话状态持久化（POC 用内存，生产接 Postgres）

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
        checkpointer=InMemorySaver(),
    )
    return graph


# 构建 agent 并暴露为 AG-UI 端点（path="/"）
_graph = build_agent()
add_langgraph_fastapi_endpoint(
    app,
    agent=LangGraphAGUIAgent(
        name="cognia_teacher",
        description="Cognia AI 主动学习教练：诊断认知盲区并动态引导掌握知识点",
        graph=_graph,
    ),
    path="/",
)


def main() -> None:
    import uvicorn

    port = int(os.getenv("AGUI_PORT", "8123"))
    uvicorn.run("cognia.server:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()
