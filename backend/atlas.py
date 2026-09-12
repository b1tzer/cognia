"""全局概念库（Cognitive Atlas）归一化匹配层。

职责：把每次 LLM 拆解出的概念，通过「规范化名」匹配到全局概念库的稳定 id，
实现跨会话累积、避免重复确认。这是「概念 id 跨会话稳定」的确定性骨架。

归一化策略（第一版，纯字符串零 token）：
- 全角转半角
- 去空白
- 英文小写
- 去常见泛化后缀（基础/机制/原理/...）

说明：第一版不引入 LLM 额外输出 canonical_name，也不做 embedding 模糊匹配
（两者均列为 Out of Scope），直接从概念名派生稳定匹配键，确定性更强、零额外 token。
"""
from __future__ import annotations

import re
import uuid

import config
import db
import learner_profile

# 常见泛化后缀：去掉后得到「核心词」，用于跨会话匹配。
# 例如「HTTP 基础」「HTTP 协议」都归一到 http。
_GENERIC_SUFFIXES = (
    "基础", "机制", "原理", "概念", "核心", "状态", "队列", "模型",
    "协议", "模式", "结构", "体系", "框架", "理论", "算法", "思想",
    "入门", "进阶", "详解", "实践", "应用", "实战", "指南", "教程",
    "介绍", "总结", "笔记",
)


def _full_to_half(s: str) -> str:
    """全角字符转半角（含全角空格）。"""
    out = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def normalize_key(name: str) -> str:
    """概念名 -> 稳定匹配键（去修饰限定 + 字符串归一）。"""
    if not name:
        return ""
    s = _full_to_half(name.strip())
    s = re.sub(r"\s+", "", s)
    # 循环去除泛化后缀（如「HTTP 协议基础」→ 先去掉「基础」再「协议」）
    changed = True
    while changed:
        changed = False
        for suf in _GENERIC_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf):
                s = s[: -len(suf)]
                changed = True
                break
    return s.lower()


def resolve_global_id(name: str, summary: str = "") -> str:
    """把概念名归一化并 upsert 到全局概念库，返回全局 id。

    命中已有概念则复用其 id（跨会话稳定）；未命中则新建（id = normalize_key）。
    任何异常都安全降级：仍返回一个可用 id，不阻塞主流程。
    """
    key = normalize_key(name)
    if not key:
        # 归一化失败（空名）：退化为带前缀的唯一 id，保证不撞库
        return f"c-{uuid.uuid4().hex[:12]}"
    try:
        db.upsert_concept(key, name, summary)
    except Exception:
        pass  # 落库失败不阻塞，id 仍可返回
    return key


def _state_from_mastery(mastery: float) -> str:
    """掌握度 -> 认知状态（与 cognitive.state_from_mastery 对齐，避免跨模块依赖）。"""
    if mastery >= config.MASTERY_THRESHOLD:
        return "understood"
    if mastery >= config.STATE_BANDS.get("partial", 0.55):
        return "partial"
    return "insufficient"


def build_atlas_view(user_id: str = db.DEFAULT_USER_ID) -> dict:
    """聚合全局概念 + 关系 + 掌握度，返回版图视图。

    - concepts：全局概念库全部概念，附 mastery/state（来自跨会话 concept_mastery，含遗忘衰减）
    - relations：概念关系（is-a/related/prerequisite）
    全局库为空时返回空列表，不报错。
    """
    concepts = [
        {
            "id": c["id"],
            "name": c["name"],
            "summary": c.get("summary", ""),
            "mastery": 0.0,
            "state": "insufficient",
        }
        for c in db.list_concepts()
    ]
    prior = learner_profile.prior_mastery(user_id)
    for c in concepts:
        if c["id"] in prior:
            c["mastery"] = round(prior[c["id"]], 4)
            c["state"] = _state_from_mastery(prior[c["id"]])
    relations = [
        {"from": r["from_id"], "to": r["to_id"], "relation_type": r["relation_type"]}
        for r in db.list_concept_relations()
    ]
    return {"concepts": concepts, "relations": relations}
