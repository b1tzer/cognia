"""Cognia 基础设施层（infrastructure）。

放与领域逻辑无关的通用底层能力（如联网搜索、外部服务接入），
与 cognia 领域层解耦。
"""

from cognia.infra.search import search_web
from cognia.infra.store import get_checkpointer, get_pool, get_store

__all__ = ["search_web", "get_checkpointer", "get_pool", "get_store"]
