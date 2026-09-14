"""Cognia 模型路由层（LangChain 抽象 + DeepSeek）。

用 LangChain 的 ChatDeepSeek 封装三个角色模型，保持模型无关（宪法 §7）：
- planner_model：知识模型构建（一次性调用，成本低）
- teacher_model：干预 / 探测生成（高频调用）
- diagnoser_model：认知诊断（核心，低 temperature + 结构化输出）

模型 ID 可通过环境变量覆盖，默认值对齐 plan §7：
- PLANNER_MODEL / TEACHER_MODEL → deepseek-v4-flash
- DIAGNOSER_MODEL → deepseek-v4-pro

注意：
- deepseek-chat / deepseek-reasoner 已于 2026-07-24 退役，本模块不再使用。
- DEEPSEEK_API_KEY 由 ChatDeepSeek 自动从环境变量读取，本模块不显式传递。
- .env 的加载（load_dotenv）由应用入口负责（任务⑦ app.py），本模块保持纯净、
  只从 os.getenv 读取，便于测试与复用。
- 「独立严格 prompt」不在此层实现：诊断 system prompt 需引用知识模型、
  当前知识点与运行时上下文，属 `diagnose` 节点的职责，由任务⑤ graph.py 构建。
  本层只负责模型实例化（model / temperature / timeout）与结构化输出绑定。
"""

import os

from langchain_core.runnables import Runnable
from langchain_deepseek import ChatDeepSeek

from cognia.schemas import Diagnosis

# ---- 默认模型 ID（对齐 plan §7）----
DEFAULT_PLANNER_MODEL = "deepseek-v4-flash"
DEFAULT_DIAGNOSER_MODEL = "deepseek-v4-pro"
DEFAULT_TEACHER_MODEL = "deepseek-v4-flash"

# ---- temperature 策略（角色语义，非环境变量覆盖）----
PLANNER_TEMPERATURE = 0.3   # 知识模型构建：需一定创造性，但整体结构化
DIAGNOSER_TEMPERATURE = 0.0  # 诊断：锁死确定性（plan §1）
TEACHER_TEMPERATURE = 0.5   # 干预/探测：需自然的教学语言


def _model_id(env_var: str, default: str) -> str:
    """从环境变量读取模型 ID，缺省用默认值。"""
    return os.getenv(env_var, default)


def get_planner_model() -> ChatDeepSeek:
    """知识模型构建模型（planner_model）。"""
    return ChatDeepSeek(
        model=_model_id("PLANNER_MODEL", DEFAULT_PLANNER_MODEL),
        temperature=PLANNER_TEMPERATURE,
        timeout=60,
        max_retries=2,
    )


def get_teacher_model() -> ChatDeepSeek:
    """干预 / 探测生成模型（teacher_model）。"""
    return ChatDeepSeek(
        model=_model_id("TEACHER_MODEL", DEFAULT_TEACHER_MODEL),
        temperature=TEACHER_TEMPERATURE,
        timeout=60,
        max_retries=2,
    )


def get_diagnoser_model() -> ChatDeepSeek:
    """认知诊断模型（diagnoser_model），低 temperature 锁死确定性。"""
    return ChatDeepSeek(
        model=_model_id("DIAGNOSER_MODEL", DEFAULT_DIAGNOSER_MODEL),
        temperature=DIAGNOSER_TEMPERATURE,
        timeout=60,
        max_retries=2,
    )


def get_structured_diagnoser() -> Runnable:
    """返回绑定 Diagnosis schema 的结构化诊断模型。

    调用后直接返回 Diagnosis 实例（Pydantic），而非 AIMessage。
    用于 diagnose 节点的结构化输出（plan §4、任务④验收标准）。
    """
    return get_diagnoser_model().with_structured_output(Diagnosis)
