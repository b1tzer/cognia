"""Cognia 联网搜索基础设施（SearXNG + 本地磁盘缓存）。

把 web_search 工具与 explain 工具共用的底层检索实现抽离到本模块，
任何联网搜索都汇聚到这里，保证检索逻辑唯一、可单点加固（超时、限流、
域名过滤等）。结果先走本地磁盘缓存（SEARCH_CACHE_TTL 秒内命中直接返回），
未命中才联网，成功后写缓存。

本模块纯标准库 + 无 LLM 依赖，可独立测试。
"""

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request

_SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", str(7 * 24 * 3600)))  # 秒，默认 7 天
_SEARCH_CACHE_RESULT_LIMIT = 10  # 缓存单 query 最多保留条数（返回时按 max_results 截断）


def _search_cache_dir() -> str:
    """搜索结果本地缓存目录（可 SEARCH_CACHE_DIR 覆盖）。"""
    env = os.getenv("SEARCH_CACHE_DIR")
    if env:
        return env
    # 本模块位于 cognia/infra/search.py，需上溯三层到项目根目录。
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "data", "search_cache")


def _normalize_search_query(query: str) -> str:
    """归一化查询词，作为缓存 key（去空白 + 统一小写）。"""
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def _search_cache_path(query: str) -> str:
    """查询词 → 本地缓存文件路径（sha1 摘要，规避非法路径字符）。"""
    digest = hashlib.sha1(_normalize_search_query(query).encode("utf-8")).hexdigest()
    return os.path.join(_search_cache_dir(), f"{digest}.json")


def _read_search_cache(query: str) -> list[dict] | None:
    """读搜索结果缓存；命中且未过期返回结果列表，否则返回 None。"""
    path = _search_cache_path(query)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    try:
        age = time.time() - float(data.get("cached_at") or -1)
    except (TypeError, ValueError):
        return None
    if age < 0 or age > _SEARCH_CACHE_TTL:
        return None
    results = data.get("results")
    return results if isinstance(results, list) else None


def _write_search_cache(query: str, results: list[dict]) -> None:
    """写搜索结果缓存（临时文件 + 原子 rename，失败不阻断主流程）。"""
    try:
        os.makedirs(_search_cache_dir(), exist_ok=True)
        tmp = _search_cache_path(query) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "query": query,
                "results": results,
                "cached_at": time.time(),
            }, f, ensure_ascii=False)
        os.replace(tmp, _search_cache_path(query))
    except OSError:
        pass


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """调用本地 SearXNG 元搜索引擎，返回 [{title, url, snippet}] 列表。

    这是 web_search 工具与 explain 工具共用的底层检索实现；任何联网搜索都
    汇聚到这里，保证检索逻辑唯一、可单点加固（超时、限流、域名过滤等）。

    结果先走本地磁盘缓存（SEARCH_CACHE_TTL 秒内命中则直接返回，避免同一
    query 反复打 SearXNG）；未命中才联网，成功后写缓存。
    """
    max_results = max(1, min(int(max_results), 10))

    # 本地缓存命中：直接返回，不再请求 SearXNG。
    cached = _read_search_cache(query)
    if cached is not None:
        return cached[:max_results]

    searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    req = urllib.request.Request(
        f"{searxng_url}/search?{params}",
        headers={"User-Agent": "cognia-agent"},
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        # 检索失败不阻断讲解，返回空列表，由上层决定降级策略。
        return []

    results = []
    for item in (data.get("results") or [])[:_SEARCH_CACHE_RESULT_LIMIT]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or "").strip()
        if not title and not url:
            continue
        results.append({
            "title": title,
            "url": url,
            "snippet": snippet[:200],
        })

    # 非空结果才落缓存（空结果缓存意义不大，且可能是临时故障）。
    if results:
        _write_search_cache(query, results)

    return results[:max_results]
