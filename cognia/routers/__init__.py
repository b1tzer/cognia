"""Cognia HTTP 路由层（除 AG-UI 接入外的辅助端点）。

与 server.py 的 AG-UI 接入层解耦：server.py 只负责 CopilotKit / LangGraph
的 AG-UI 协议端点，其余业务查询（知识星图聚合）与会话管理（/threads CRUD）
按领域拆成独立 router，经 `app.include_router(...)` 挂载。
"""

from cognia.routers.knowledge_map import router as knowledge_map_router
from cognia.routers.threads import router as threads_router
from cognia.routers.wiki import router as wiki_router

__all__ = ["knowledge_map_router", "threads_router", "wiki_router"]
