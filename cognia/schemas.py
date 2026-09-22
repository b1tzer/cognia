"""Cognia 数据模型层。

定义认知状态、置信度等核心领域模型（Pydantic Schema）。
与 spec v2.0 五态定义、plan.md §4 数据模型一一对应。

注意分层：本模块只放「纯数据模型」，算法逻辑（BKT 融合等）放 proficiency_engine.py。
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CognitiveState(str, Enum):
    """认知状态五态（spec v2.0）。"""

    UNASSESSED = "unassessed"        # 未评估（初始态）
    MASTERED = "mastered"            # 已掌握
    PARTIAL = "partial"              # 部分掌握
    MISCONCEPTION = "misconception"  # 错误理解
    UNKNOWN = "unknown"              # 盲区（确认不知道）


class Confidence(str, Enum):
    """置信度三级（clarifications Q4：剔除硬编码百分比，只做行为分级）。"""

    HIGH = "high"      # 证据充分
    MEDIUM = "medium"  # 存在歧义
    LOW = "low"        # 证据不足

class BloomLevel(str, Enum):
    """认知层级（布鲁姆分类，派生验证深度 verification_depth）。"""

    REMEMBER = "remember"        # 记忆
    UNDERSTAND = "understand"    # 理解
    APPLY = "apply"              # 应用
    ANALYZE = "analyze"          # 分析
    EVALUATE = "evaluate"        # 评价
    CREATE = "create"            # 创造


class PointType(str, Enum):
    """知识点类型（决定教学与验证方式）。"""

    CONCEPT = "concept"          # 概念
    FACT = "fact"                # 事实
    SKILL = "skill"              # 技能
    PRINCIPLE = "principle"      # 原理
    PROCESS = "process"          # 流程


class PointAttributes(BaseModel):
    """知识点本体属性（只描述知识本身，与「谁学得怎么样」无关）。"""

    type: PointType                              # 类型
    difficulty: int = Field(ge=1, le=5)          # 难度 1-5（影响 BKT 参数）
    importance: int = Field(ge=1, le=5)          # 重要度 1-5（决定教学优先级）
    bloom_level: BloomLevel                        # 认知层级（派生 verification_depth）


class KnowledgePoint(BaseModel):
    """知识模型中的单个知识点。"""

    id: str
    name: str
    description: str
    prerequisites: list[str] = Field(default_factory=list)  # 依赖的知识点 id
    attributes: PointAttributes | None = None                # 本体属性（可选，避免破坏既有调用）


class KnowledgeModel(BaseModel):
    """知识模型：学习目标 + 知识点列表。"""

    goal: str
    points: list[KnowledgePoint] = Field(default_factory=list)


def _coerce_str_list(v):
    """把 LLM 可能误输出的字符串归一化为 list[str]。

    背景：structured_output 的降级路径（models.invoke_structured）下，模型有时把
    list[str] 字段输出成 JSON 字符串（如 evidence: '["..."]'）而非数组，导致
    model_validate 抛 ValidationError。这是数据契约的容错归一化，不是约束模型行为。
    """
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        return [s]
    if isinstance(v, list):
        return [str(x) for x in v]
    return v


class Diagnosis(BaseModel):
    """认知诊断结果（候选，非最终迁移结果）。

    重要：state-machine §5 通用规则 1——诊断是「候选」，状态迁移是「裁决」。
    `diagnosis.state` 绝不直接写入长期 Proficiency。
    """

    point_id: str
    state: CognitiveState
    confidence: Confidence
    evidence: list[str] = Field(default_factory=list)  # 用户原话片段，严禁脑补（spec §6）

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, v):
        return _coerce_str_list(v)


class Observation(BaseModel):
    """AI 对用户理解程度的一次观察样本（只追加，非最终结论）。

    这是知识版图子系统的「过程数据」：AI 每次判断用户对某知识点的理解程度，
    只提交一条 Observation，系统用 BKT 算法融合后才产出权威 Proficiency。
    AI 无权直接改写权威状态，只能提交观察值。
    """

    point_id: str
    observed_state: CognitiveState                    # AI 判定的五态（观察值，非结论）
    confidence: Confidence                             # AI 的置信度
    evidence: list[str] = Field(default_factory=list)  # 用户原话片段，严禁脑补（spec §6）
    observer: Literal["ai"] = "ai"                     # 观察者身份（预留多源）
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("evidence", mode="before")
    @classmethod
    def _normalize_evidence(cls, v):
        return _coerce_str_list(v)


class Proficiency(BaseModel):
    """系统产出的权威熟练度（AI 只读，由 BKT 融合观察值计算得出）。

    关键区分：`mapped_state` 是系统权威状态，与 `Observation.observed_state`
    （AI 判了什么）严格不同。AI 只能提交观察值并查询本结果，无权直接改写。
    """

    point_id: str
    latent_value: float = Field(ge=0.0, le=1.0)  # 连续值 P(learned)，BKT 后验 ∈ [0,1]
    mapped_state: CognitiveState                    # 离散化后的五态（唯一对外权威状态）
    uncertainty: float = Field(ge=0.0, le=1.0)     # 不确定性（1 - max(p, 1-p)）
    source_algorithm: str = "bkt"                   # 来源算法（当前仅 BKT，预留扩展）
    observation_count: int = 0                      # 已融合的有效观察次数
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---- 个人 Wiki（第三资产：人可读知识长文）----

class WikiAuthor(str, Enum):
    """wiki 页面最近一次写入者（决定 AI 是否可覆盖用户手改）。"""

    AI = "ai"        # AI 生成的草稿
    USER = "user"    # 用户手改（AI 只追加新版本、不覆盖）


class WikiPage(BaseModel):
    """个人 wiki 的一页：人可读、可在线编辑的知识长文。

    正文（content_markdown）存 Git 仓库（版本真相），本模型作为元数据契约，
    正文 + 溯源字段共同构成一个 wiki 页面。路线 B：wiki 有独立页面树
    （parent_page_id / path），不复用知识版图 DAG。
    """

    page_id: str                                  # 稳定 id（slug）
    title: str
    content_markdown: str = ""                    # 正文（存 Git 文件）
    path: str = ""                                # 页面树路径（独立体系）
    tags: list[str] = Field(default_factory=list)
    parent_page_id: str | None = None
    author: WikiAuthor = WikiAuthor.AI            # 最近一次写入者
    source_thread_id: str | None = None           # AI 生成时来源会话
    source_turns: list[int] = Field(default_factory=list)  # 来源轮次（1-based）
    evidence: list[str] = Field(default_factory=list)      # 溯源证据（用户原话）
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WikiRevision(BaseModel):
    """wiki 页面的一个版本（对应一次 git commit）。"""

    page_id: str
    commit_hash: str
    author: WikiAuthor = WikiAuthor.AI
    summary: str = ""                             # 变更说明
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WikiSummary(BaseModel):
    """对话总结为 wiki 草稿的 LLM 结构化输出（供 summarize 提炼用）。"""

    title: str
    slug: str                                     # 英文小写连字符，作为 page_id
    markdown: str                                 # 正文
    evidence: list[str] = Field(default_factory=list)  # 溯源证据（用户原话，严禁脑补）
    tags: list[str] = Field(default_factory=list)      # 英文小写标签

    @field_validator("evidence", "tags", mode="before")
    @classmethod
    def _normalize_lists(cls, v):
        return _coerce_str_list(v)
