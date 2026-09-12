"""Pydantic 数据模型：定义 API 请求/响应结构与内部实体。"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

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
    related: list[str] = Field(default_factory=list)  # 横向相关概念名列表（LLM 额外输出，落库为 related 关系）

    @field_validator("prerequisites", "common_misconceptions", "related", mode="before")
    @classmethod
    def _coerce_to_list(cls, v):
        """LLM 可能把列表字段返回成字符串，这里统一转成列表。"""
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            return [v] if v else []
        if isinstance(v, list):
            return [str(x) for x in v if x]
        return [str(v)] if v else []


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
    consecutive_failures: int = 0 # 连续失败次数（用于回溯触发）
    last_evidence: str = ""       # 最近一次证据简述


class CognitiveModel(BaseModel):
    goal: str
    concepts: list[ConceptMastery] = Field(default_factory=list)
    updated_at: str = ""
    profile: Optional[dict] = None   # 用户画像（千人千面）：level/label/params


# ---------------------------------------------------------------------------
# 对话 / 消息
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    action: Optional[TutorAction] = None
    diagnosis: Optional[dict] = None   # 本轮诊断结果摘要
    clarify: Optional[dict] = None     # 澄清信息（candidates/question，仅澄清阶段消息携带）


class Session(BaseModel):
    id: str
    goal: str
    created_at: str
    messages: list[Message] = Field(default_factory=list)
    cognitive: Optional[CognitiveModel] = None
    knowledge: Optional[KnowledgeModel] = None
    status: Literal["active", "completed"] = "active"
    # 生命周期阶段：clarifying（澄清目标中，知识模型未建）→ active（学习中）→ completed（已完成）
    stage: Literal["clarifying", "active", "completed"] = "active"


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

class ActionReason(BaseModel):
    """教学动作选择的结构化理由（用于可解释性与程序化校验）。"""
    evidence_cited: str = ""      # 引用的诊断证据字段
    criterion_used: str = ""      # 采用的判别准则
    pedagogical_intent: str = ""  # 教学意图
    confidence: float = 0.0       # 决策置信度 [0,1]

class ActionDecision(BaseModel):
    """教学决策层输出：从候选集内选定的动作 + 理由。"""
    chosen_action: TutorAction = "probe"
    reasons: ActionReason = Field(default_factory=ActionReason)

class FocusDecision(BaseModel):
    """焦点概念决策层输出：选定的下一个焦点概念。"""
    selected_concept_id: str = ""
    is_backtrack: bool = False
    backtrack_target_id: Optional[str] = None
    reason: str = ""
    confidence: float = 0.0
