"""Cognia 可观测性开关（Langfuse，开源 MIT，可选）。

设计目标（见需求 issue「Cognia 接入 Langfuse 可观测」）：
- **零侵入默认关闭**：仅当 `.env` 的 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`
  / `LANGFUSE_HOST` 三变量**齐全**时才构建 Langfuse `CallbackHandler`，否则
  `LANGFUSE_HANDLER is None`，调用方不注入任何 callback，生产无感知、无网络请求。
- **request-time callback 向下传播**：LangChain 把 `config["callbacks"]` 自动透传
  到所有子 runnable（含 tool 节点），因此主 agent 每轮 LLM + 工具节点 span
  零代码自动记录；工具内部的核心子 LLM 调用需手动透传 `config` 才能挂到 trace 树。

用法（server.py）：
    from cognia.observability import LANGFUSE_HANDLER
    if LANGFUSE_HANDLER is not None:
        config["callbacks"] = [LANGFUSE_HANDLER]
        config["metadata"] = {"langfuse_user_id": ..., "langfuse_session_id": ...}
"""

import os

# 三变量齐全时构建 handler，否则为 None。导入本模块零副作用（不连接网络）。
_LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
_LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
_LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")

if _LANGFUSE_PUBLIC_KEY and _LANGFUSE_SECRET_KEY and _LANGFUSE_HOST:
    try:
        from langfuse.langchain import CallbackHandler

        LANGFUSE_HANDLER: "CallbackHandler | None" = CallbackHandler(
            public_key=_LANGFUSE_PUBLIC_KEY,
            secret_key=_LANGFUSE_SECRET_KEY,
            host=_LANGFUSE_HOST,
        )
    except Exception as exc:  # 导入/初始化失败不阻断主流程
        print(f"[Cognia] Langfuse 初始化失败，已关闭可观测：{exc}")
        LANGFUSE_HANDLER = None
else:
    LANGFUSE_HANDLER = None
