"""Cognia 策略性 prompt 统一入口（prompts-as-code）。

把决定 AI 行为的 system prompt 从业务逻辑代码中抽离到本包，
独立成文件，便于 diff / review / 版本化与回归测试。

各文件对应原模块：
- teacher.py：教学 Agent
- learning_engine.py：诊断器 / 知识建模器（原 learning_engine.py）
"""

from cognia.prompts.teacher import (
    EXPLAINER_SYSTEM_PROMPT,
    PROBE_GENERATOR_SYSTEM_PROMPT,
    TEACHER_SYSTEM_PROMPT,
)
from cognia.prompts.learning_engine import (
    DIAGNOSER_SYSTEM_PROMPT,
    KNOWLEDGE_MODELER_SYSTEM_PROMPT,
)

__all__ = [
    "TEACHER_SYSTEM_PROMPT",
    "PROBE_GENERATOR_SYSTEM_PROMPT",
    "EXPLAINER_SYSTEM_PROMPT",
    "DIAGNOSER_SYSTEM_PROMPT",
    "KNOWLEDGE_MODELER_SYSTEM_PROMPT",
]
