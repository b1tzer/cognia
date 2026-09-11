"""Pydantic 数据模型：定义 API 请求/响应结构与内部实体。"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# 认知状态四分类（对应 Purpose.md）
CognitiveState = Literal["understood", "partial", "misconceived", "insufficient"]

# 教学动作（对应 Purpose.md）
TutorAction = Literal["probe", "explain", "correct", "backtrack", "advance"]


# ---------------------------------------------------------------------------
# 知识模型
# ---------------------------------------------------------------------------
class Concept(BaseModel):
    """知识图谱中的一个概念节点。"""
    id: str
    name: str
    summary: str = ""           # 一句话说明该概念是什么
    why_matters: str = ""       # 真正理解它意味着什么 / 为什么重要
    prerequisites: list[str] = Field(default_factory=list)  # 前置概念 id 列表
    common_misconceptions: list[str] = Field(default_factory=list)


class KnowledgeModel(BaseModel):
    """由学习目标推导出的完整知识模型（DAG）。"""
    goal: str
    root_concepts: list[str] = Field(default_factory=list)  # 顶层目标概念
    concepts: list[Concept] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 认知模型（学生模型）
# ---------------------------------------------------------------------------
class ConceptMastery(BaseModel):
    concept_id: str
    concept_name: str
    mastery: float = 0.0          # 掌握概率 [0,1]
    state: CognitiveState = "insufficient"
    evidence_count: int = 0       # 已收集的证据条数
    last_evidence: str = ""       # 最近一次证据简述


class CognitiveModel(BaseModel):
    goal: str
    concepts: list[ConceptMastery] = Field(default_factory=list)
    updated_at: str = ""


# ---------------------------------------------------------------------------
# 对话 / 消息
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    action: Optional[TutorAction] = None
    diagnosis: Optional[dict] = None   # 本轮诊断结果摘要


class Session(BaseModel):
    id: str
    goal: str
    created_at: str
    messages: list[Message] = Field(default_factory=list)
    cognitive: Optional[CognitiveModel] = None
    knowledge: Optional[KnowledgeModel] = None
    status: Literal["active", "completed"] = "active"


# ---------------------------------------------------------------------------
# API 请求/响应
# ---------------------------------------------------------------------------
class StartSessionRequest(BaseModel):
    goal: str


class ChatRequest(BaseModel):
    content: str


class DiagnosticResult(BaseModel):
    """诊断引擎对一次用户表达的输出。"""
    state: CognitiveState
    confidence: float
    concept_ids: list[str] = Field(default_factory=list)
    evidence: str = ""
    misconception: str = ""
    missing: list[str] = Field(default_factory=list)
