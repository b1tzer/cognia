"""Cognia 概念语义合并引擎（跨对话概念身份归一）。

把「新知识点 name」与「用户已有概念」做语义比对，判定是否同一概念，
返回全局稳定 id。核心目标：以 user 为单位构建全局知识星图，而非以对话为单位。

分层（对齐宪法）：
- embedding 是特征提取，cosine 数值比较是「记忆点」用确定性数学。
- 合并阈值是「策略参数」（env 可覆盖）。
- 本模块纯函数、零 LLM、零 IO，可被内存数据直接驱动测试。

关键设计：
- 全局 id = concept:{sha1(归一化 name)[:8]}，首次出现的 name 为 canonical_name。
- embedding 可用：新 name 与存量概念 embedding 全量 cosine 比对，≥ 阈值则复用。
- embedding 不可用：退化为归一化 name 精确匹配（同名才合并）。
"""

import hashlib
import os
import re

from cognia import embedding

DEFAULT_THRESHOLD = 0.85


def normalize_name(name: str) -> str:
    """归一化概念名（去首尾空白 + 折叠内部空白 + 统一小写）。"""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def concept_id(name: str) -> str:
    """由 canonical name 生成全局稳定 id。"""
    digest = hashlib.sha1(normalize_name(name).encode("utf-8")).hexdigest()[:8]
    return f"concept:{digest}"


def _threshold(threshold: float | None) -> float:
    """解析合并阈值（显式参数优先，否则 env，再否则默认值）。"""
    if threshold is not None:
        return float(threshold)
    raw = os.getenv("CONCEPT_MERGE_THRESHOLD", str(DEFAULT_THRESHOLD))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD


def merge_concept(
    name: str,
    existing: list[dict],
    threshold: float | None = None,
) -> tuple[str, bool, str]:
    """判定 name 归属哪个全局概念，返回 (global_id, is_new, canonical_name)。

    - existing：存量概念列表，每项 `{"id": str, "name": str, "embedding": list[float] | None}`。
    - threshold：合并阈值，默认 DEFAULT_THRESHOLD，可 env 覆盖。
    - is_new=True 表示新建概念，global_id 为新建 id；否则复用已有概念的 id。
    """
    th = _threshold(threshold)
    vec = embedding.embed_text(name)

    if vec is not None:
        # 语义合并：与存量 embedding 全量 cosine 比对，取最高分且 ≥ 阈值者
        best = None
        best_score = -1.0
        for c in existing:
            cv = c.get("embedding")
            if not cv:
                continue
            s = embedding.cosine(vec, cv)
            if s > best_score:
                best_score = s
                best = c
        if best is not None and best_score >= th:
            return best["id"], False, best.get("name") or name

    # 确定性兜底：归一化 name 精确匹配（同名才合并）
    target = normalize_name(name)
    for c in existing:
        if normalize_name(c.get("name", "")) == target:
            return c["id"], False, c.get("name") or name

    # 新建概念
    return concept_id(name), True, name
