"""Cognia 数据模型层。

定义认知状态、置信度、验证结果等核心领域模型（Pydantic Schema）。
与 state-machine.md 五态定义、plan.md §4 数据模型一一对应。

注意分层：本模块只放「纯数据模型」，状态迁移逻辑（can_transition 等）放 state_machine.py（任务③）。
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


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


class ValidationResult(str, Enum):
    """单项验证结果（区分「未评估」与「验证失败」，避免 bool 混淆）。"""

    UNASSESSED = "unassessed"  # 尚未验证
    PASSED = "passed"          # 验证通过
    FAILED = "failed"          # 验证失败


class KnowledgePoint(BaseModel):
    """知识模型中的单个知识点。"""

    id: str
    name: str
    description: str
    prerequisites: list[str] = Field(default_factory=list)  # 依赖的知识点 id


class KnowledgeModel(BaseModel):
    """知识模型：学习目标 + 知识点列表。"""

    goal: str
    points: list[KnowledgePoint] = Field(default_factory=list)


class Diagnosis(BaseModel):
    """认知诊断结果（候选，非最终迁移结果）。

    重要：state-machine §5 通用规则 1——诊断是「候选」，状态迁移是「裁决」。
    `diagnosis.state` 绝不直接写入长期 Proficiency。
    """

    point_id: str
    state: CognitiveState
    confidence: Confidence
    evidence: list[str] = Field(default_factory=list)  # 用户原话片段，严禁脑补（spec §6）


class VerificationState(BaseModel):
    """双重验证状态（plan §3.5：mastered 判定需「概念解释 + 场景辨析」双过）。"""

    concept: ValidationResult = ValidationResult.UNASSESSED    # 概念解释验证结果
    scenario: ValidationResult = ValidationResult.UNASSESSED   # 场景 / 反例辨析验证结果
    concept_evidence: list[str] = Field(default_factory=list)  # 概念验证证据（用户原话）
    scenario_evidence: list[str] = Field(default_factory=list)  # 场景验证证据（用户原话）
    current_step: Literal["concept", "scenario", "done"] = "concept"  # 当前验证到哪一步


class ProficiencyEntry(BaseModel):
    """熟练度条目（增量 Delta，禁止全量重写，宪法 §5）。"""

    point_id: str
    from_state: CognitiveState | None = None  # 状态迁移起点（首次诊断时为 None）
    to_state: CognitiveState                   # 状态迁移终点
    evidence: list[str] = Field(default_factory=list)  # 支撑本次迁移的用户原话（非 AI 总结）
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    update_type: Literal["delta"] = "delta"   # 增量 Delta，严禁全量重写


class Intervention(BaseModel):
    """主动干预动作（追问 / 解释 / 纠错 / 回溯）。"""

    point_id: str
    intervention_type: Literal["probe", "explain", "correct", "backtrack"]
    content: str
