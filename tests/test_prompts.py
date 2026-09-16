"""prompt 抽离后的最小回归测试（prompts-as-code）。

验证两类契约：
1. 抽离到 cognia.prompts 的 system prompt 非空，且保留关键「铁律」指令
   （防止有人误删决定 AI 行为的关键约束）；
2. 业务模块（learning_engine）从 prompts 包 import 的常量与权威文本一致，
   防止未来复制粘贴导致两处文本漂移。
"""

from cognia import prompts
from cognia.prompts.learning_engine import (
    CONCEPT_VERIFIER_SYSTEM_PROMPT,
    DIAGNOSER_SYSTEM_PROMPT,
    KNOWLEDGE_MODELER_SYSTEM_PROMPT,
    SCENARIO_VERIFIER_SYSTEM_PROMPT,
)
from cognia.prompts.teacher import TEACHER_SYSTEM_PROMPT


# ---- 1. 关键 prompt 非空 ----

def test_all_system_prompts_non_empty():
    """五个策略性 system prompt 均非空。"""
    for prompt in (
        TEACHER_SYSTEM_PROMPT,
        DIAGNOSER_SYSTEM_PROMPT,
        KNOWLEDGE_MODELER_SYSTEM_PROMPT,
        CONCEPT_VERIFIER_SYSTEM_PROMPT,
        SCENARIO_VERIFIER_SYSTEM_PROMPT,
    ):
        assert isinstance(prompt, str) and prompt.strip()


# ---- 2. 关键铁律存在（防误删决定行为的关键约束）----

def test_teacher_prompt_keeps_diagnosis_guardrail():
    """教学 Agent 不得直接改掌握状态，只能提交观察值，由系统 BKT 定级。"""
    assert "不能直接判定或修改学生的掌握状态" in TEACHER_SYSTEM_PROMPT
    assert "propose_diagnosis" in TEACHER_SYSTEM_PROMPT
    assert "record_observation" in TEACHER_SYSTEM_PROMPT
    assert "query_proficiency" in TEACHER_SYSTEM_PROMPT


def test_teacher_prompt_keeps_web_search_guardrail():
    """讲解专业知识前必须联网验证技术事实。"""
    assert "web_search" in TEACHER_SYSTEM_PROMPT
    assert "验证技术事实" in TEACHER_SYSTEM_PROMPT


def test_diagnoser_prompt_keeps_evidence_rule():
    """诊断证据必须来自用户原话，严禁脑补（spec §6）。"""
    assert "证据必须来自用户原话片段" in DIAGNOSER_SYSTEM_PROMPT
    assert "严禁脑补" in DIAGNOSER_SYSTEM_PROMPT


def test_diagnoser_prompt_defines_five_states():
    """五态定义完整（spec v2.0）。"""
    for state in ("unassessed", "unknown", "partial", "misconception", "mastered"):
        assert state in DIAGNOSER_SYSTEM_PROMPT


def test_verifier_prompts_keep_role():
    """三个认知层辅助 prompt 保留各自角色定位。"""
    assert "知识建模器" in KNOWLEDGE_MODELER_SYSTEM_PROMPT
    assert "概念解释" in CONCEPT_VERIFIER_SYSTEM_PROMPT
    assert "场景" in SCENARIO_VERIFIER_SYSTEM_PROMPT


# ---- 3. 业务模块 import 的就是 prompts 包权威常量（防复制漂移）----

def test_learning_engine_import_matches_prompts_package():
    """learning_engine 引入的常量与 prompts 包权威文本一致。"""
    from cognia import learning_engine
    assert learning_engine.DIAGNOSER_SYSTEM_PROMPT == DIAGNOSER_SYSTEM_PROMPT
    assert learning_engine.KNOWLEDGE_MODELER_SYSTEM_PROMPT == KNOWLEDGE_MODELER_SYSTEM_PROMPT
    assert learning_engine.CONCEPT_VERIFIER_SYSTEM_PROMPT == CONCEPT_VERIFIER_SYSTEM_PROMPT
    assert learning_engine.SCENARIO_VERIFIER_SYSTEM_PROMPT == SCENARIO_VERIFIER_SYSTEM_PROMPT


def test_prompts_package_exports_all():
    """prompts 包统一导出全部五个 system prompt。"""
    for name in (
        "TEACHER_SYSTEM_PROMPT",
        "DIAGNOSER_SYSTEM_PROMPT",
        "KNOWLEDGE_MODELER_SYSTEM_PROMPT",
        "CONCEPT_VERIFIER_SYSTEM_PROMPT",
        "SCENARIO_VERIFIER_SYSTEM_PROMPT",
    ):
        assert hasattr(prompts, name)
