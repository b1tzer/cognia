"""知识模型构建器：把学习目标转换为概念依赖图（DAG）。

优先使用 LLM 生成精准概念图；未配置模型时回退到内置领域模板
与通用学习框架，保证对任意学习目标都能构建出可用的知识模型。
"""
from __future__ import annotations

import re
from typing import Any

from llm import chat_json
import prompt_rules
import db
from schemas import Concept, KnowledgeModel
import atlas

# ---------------------------------------------------------------------------
# LLM 提示词
# ---------------------------------------------------------------------------
_BUILD_SYSTEM = """你是资深课程设计师。把学习目标拆成一张概念依赖图(DAG)。

严格按以下 JSON 输出，不要任何解释或思考过程，尽量简短：
{"root_concepts":["id"],"concepts":[{"id":"english-id","name":"中文名","summary":"一句话","why_matters":"一句话","prerequisites":["id"],"common_misconceptions":["一句话"],"related":["相关概念名"]}]}

规则：
1. concepts 共 4~7 个概念，覆盖从基础到目标。
2. prerequisites 引用 concepts 中已出现的 id，构成 DAG，不能成环。
3. root_concepts 是最终要掌握的核心概念 id。
4. 每个字符串字段尽量一句话，不要展开。
5. 必须完整保留目标中的修饰限定词（如「高并发 IO」的「高并发」），拆解出的概念必须围绕完整目标，不能只取名词主干（不能把「高并发 IO」简化成「IO 基础」）。
6. related 是横向相关但非前置依赖的概念名（可为 concepts 之外的概念名，也可留空），用于版图周边关联。
"""


def _build_with_llm(goal: str) -> KnowledgeModel | None:
    data = chat_json(_BUILD_SYSTEM + prompt_rules.rules_suffix("domain_model"), f"学习目标：{goal}", temperature=0.3, budget_label="domain_model")
    if not data:
        return None
    # 逐条校验，跳过缺失 id/name 等无效节点，提升对不稳定输出的容错
    concepts: list[Concept] = []
    for c in data.get("concepts", []):
        try:
            concepts.append(Concept(**c))
        except Exception:
            continue
    if not concepts:
        return None
    valid_ids = {c.id for c in concepts}
    root = [r for r in data.get("root_concepts", []) if r in valid_ids] or [concepts[-1].id]
    return KnowledgeModel(goal=goal, root_concepts=root, concepts=concepts)


# ---------------------------------------------------------------------------
# 内置领域概念图（降级模板）
# ---------------------------------------------------------------------------
_DOMAIN_TEMPLATES: dict[str, list[Concept]] = {
    "http": [
        Concept(id="http-basics", name="HTTP 基础", summary="HTTP 是无状态的请求-响应应用层协议",
                why_matters="理解请求/响应的本质是读懂一切 Web 交互的前提",
                prerequisites=[], common_misconceptions=["认为 HTTP 是加密的", "以为一个连接只能传一个请求"]),
        Concept(id="request-response", name="请求与响应结构", summary="方法、URL、头部、状态码、消息体",
                why_matters="能读懂抓包、定位接口问题的关键", prerequisites=["http-basics"],
                common_misconceptions=["分不清 header 和 body 的职责"]),
        Concept(id="tls", name="HTTPS 与 TLS 握手", summary="TLS 协商加密与身份验证，承载 HTTP",
                why_matters="理解安全通信的信任链", prerequisites=["http-basics"],
                common_misconceptions=["认为 HTTPS 隐藏了所有信息（其实域名仍明文）"]),
        Concept(id="caching", name="缓存与协商", summary="Cache-Control、ETag、Last-Modified",
                why_matters="性能优化的核心手段", prerequisites=["request-response"],
                common_misconceptions=["认为缓存一定更快，忽略失效一致性"]),
        Concept(id="session", name="会话与状态保持", summary="Cookie、Session、Token 如何在无状态协议上保持状态",
                why_matters="理解登录态与鉴权的基础", prerequisites=["request-response"],
                common_misconceptions=["认为 Cookie 一定是安全的"]),
        Concept(id="http2-3", name="HTTP/2 与 HTTP/3", summary="多路复用、头部压缩、QUIC",
                why_matters="理解现代 Web 性能与演进方向", prerequisites=["http-basics", "tls"],
                common_misconceptions=["认为 HTTP/2 一定比 HTTP/1.1 快"]),
    ],
    "git": [
        Concept(id="snapshot", name="快照与提交", summary="Git 以内容寻址的快照记录版本，而非差异",
                why_matters="理解 Git 数据模型是掌握一切命令的钥匙", prerequisites=[],
                common_misconceptions=["认为 Git 保存的是文件差异"]),
        Concept(id="working-tree", name="工作区/暂存区/仓库", summary="三个区域的状态流转",
                why_matters="理解 add/commit 到底做了什么", prerequisites=["snapshot"],
                common_misconceptions=["把暂存区当成备份"]),
        Concept(id="branch", name="分支与引用", summary="分支是指向提交的轻量指针",
                why_matters="理解分支的本质才能灵活协作", prerequisites=["snapshot"],
                common_misconceptions=["认为分支是代码副本"]),
        Concept(id="merge-rebase", name="合并与变基", summary="merge 与 rebase 的差异和适用场景",
                why_matters="团队协作中保持历史清晰的关键", prerequisites=["branch"],
                common_misconceptions=["认为 rebase 会丢失历史"]),
        Concept(id="remote", name="远程与协作", summary="remote、fetch/pull/push 的协作模型",
                why_matters="多人协作与冲突处理的基础", prerequisites=["branch"],
                common_misconceptions=["分不清 fetch 与 pull"]),
    ],
    "python": [
        Concept(id="data-model", name="对象与引用", summary="Python 一切皆对象，变量是名字绑定",
                why_matters="理解可变/不可变与传参行为的根基", prerequisites=[],
                common_misconceptions=["以为赋值是复制", "混淆可变与不可变"]),
        Concept(id="control-flow", name="控制流", summary="条件、循环、推导式",
                why_matters="程序逻辑的基础构建块", prerequisites=["data-model"],
                common_misconceptions=["滥用推导式牺牲可读性"]),
        Concept(id="function-scope", name="函数与作用域", summary="def、参数、闭包、LEGB 作用域",
                why_matters="组织代码与理解变量的基础", prerequisites=["data-model"],
                common_misconceptions=["混淆全局变量与闭包捕获"]),
        Concept(id="oop", name="类与对象", summary="类、实例、继承、多态、魔术方法",
                why_matters="建模复杂系统的核心工具", prerequisites=["function-scope"],
                common_misconceptions=["以为继承总是优于组合"]),
        Concept(id="exception", name="异常处理", summary="try/except、异常层级、上下文管理",
                why_matters="编写健壮程序的关键", prerequisites=["function-scope"],
                common_misconceptions=["捕获所有异常且静默吞掉"]),
    ],
}


def _generic_concepts(goal: str) -> list[Concept]:
    """通用学习框架，兜底任意目标。"""
    topic = goal.strip()
    return [
        Concept(id="foundation", name=f"{topic}·基础概念", summary="该领域最核心的术语与定义",
                why_matters="没有准确术语，后续推理无法建立", prerequisites=[],
                common_misconceptions=["用日常直觉替代精确定义"]),
        Concept(id="mechanism", name="核心原理/机制", summary="事物运作的内在机制与因果链",
                why_matters="从'知道是什么'到'理解为什么'的跨越", prerequisites=["foundation"],
                common_misconceptions=["只记结论，不理解推导"]),
        Concept(id="components", name="关键组成/要素", summary="构成整体的关键部分及其关系",
                why_matters="建立结构化的心智模型", prerequisites=["mechanism"],
                common_misconceptions=["孤立记忆部件，忽略相互作用"]),
        Concept(id="application", name="应用与场景", summary="在真实场景中如何运用与迁移",
                why_matters="检验是否真正理解、能否迁移的试金石", prerequisites=["components"],
                common_misconceptions=["会做题等于会应用"]),
        Concept(id="pitfalls", name="常见误区与边界", summary="容易出错的地方与适用范围",
                why_matters="避免在边界条件下犯错", prerequisites=["foundation"],
                common_misconceptions=["把特例当规律"]),
    ]


def _normalize_km(km: KnowledgeModel) -> KnowledgeModel:
    """把概念 id 归一化为全局稳定 id，并同步替换 prerequisites / root_concepts 引用。

    这是「概念 id 跨会话稳定」的落点：相同语义概念归一到同一全局 id，
    使跨会话的 concept_mastery 先验能命中，避免重复确认。
    """
    id_map: dict[str, str] = {}
    for c in km.concepts:
        id_map[c.id] = atlas.resolve_global_id(c.name, c.summary)

    for c in km.concepts:
        c.id = id_map.get(c.id, c.id)
        c.prerequisites = [id_map.get(p, p) for p in c.prerequisites]
    km.root_concepts = [id_map.get(r, r) for r in km.root_concepts]
    return km


def _persist_relations(km: KnowledgeModel) -> None:
    """把 DAG 边（prerequisite）与 related 概念名落库为概念关系（幂等）。

    - prerequisite：from=前置概念，to=后继概念（学 to 前需先学 from）。
    - related：横向相关，from=当前概念，to=归一化后的相关概念（LLM 输出的名字）。
    任一异常安全降级：关系落库失败不阻塞知识模型构建。
    """
    ids = {c.id for c in km.concepts}
    try:
        for c in km.concepts:
            for p in c.prerequisites:
                if p in ids and p != c.id:
                    db.upsert_concept_relation(p, c.id, "prerequisite")
            for rname in c.related:
                rid = atlas.resolve_global_id(rname)
                if rid and rid != c.id:
                    db.upsert_concept_relation(c.id, rid, "related")
    except Exception:
        pass


def build_knowledge_model(goal: str) -> KnowledgeModel:
    """构建知识模型：LLM 优先，降级到内置模板；概念 id 统一归一化为全局稳定 id。"""
    km = _build_with_llm(goal)
    if km is None:
        low = goal.lower()
        concepts: list[Concept] | None = None
        for key in ("http", "git", "python", "机器学习", "machine learning"):
            if key in low:
                concepts = _DOMAIN_TEMPLATES.get(key if key in _DOMAIN_TEMPLATES else "http")
                break

        if concepts is None:
            concepts = _generic_concepts(goal)

        # 兜底：若内置模板无匹配，仍用通用模板
        if not concepts:
            concepts = _generic_concepts(goal)

        root = [concepts[-1].id] if concepts else []
        km = KnowledgeModel(goal=goal, root_concepts=root, concepts=concepts)

    km = _normalize_km(km)
    _persist_relations(km)
    return km
