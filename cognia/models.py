"""Cognia 模型路由层（LangChain 抽象 + DeepSeek）。

用 LangChain 的 ChatDeepSeek 封装三个角色模型，保持模型无关（宪法 §7）：
- planner_model：知识模型构建（一次性调用，成本低）
- teacher_model：干预 / 探测生成（高频调用）
- diagnoser_model：认知诊断（核心，低 temperature + 结构化输出）

模型 ID 可通过环境变量覆盖，默认值对齐 plan §7：
- PLANNER_MODEL / TEACHER_MODEL → deepseek-v4-flash
- DIAGNOSER_MODEL → deepseek-v4-flash

注意：
- deepseek-chat / deepseek-reasoner 已于 2026-07-24 退役，本模块不再使用。
- LLM_API_BASE / LLM_API_KEY 由本模块从环境变量读取，指向本地 OpenAI 兼容 adapter。
- .env 的加载（load_dotenv）由应用入口负责（任务⑦ app.py），本模块保持纯净、
  只从 os.getenv 读取，便于测试与复用。
- 「独立严格 prompt」不在此层实现：诊断 system prompt 需引用知识模型、
  当前知识点与运行时上下文，属 `diagnose` 节点的职责，由任务⑤ graph.py 构建。
  本层只负责模型实例化（model / temperature / timeout）与结构化输出绑定。
"""

import json
import os

from langchain_core.runnables import Runnable
from langchain_deepseek import ChatDeepSeek

from cognia.schemas import Diagnosis

# ---- 默认模型 ID（对齐 plan §7）----
DEFAULT_PLANNER_MODEL = "deepseek-v4-flash"
DEFAULT_DIAGNOSER_MODEL = "deepseek-v4-flash"
DEFAULT_TEACHER_MODEL = "deepseek-v4-flash"

# ---- 本地 OpenAI 兼容 adapter 接入（默认指向本机 adapter）----
# /opt/tdai/adapter/adapter.py 监听 127.0.0.1:8090，把标准 OpenAI 请求转发到工蜂 Gateway。
DEFAULT_API_BASE = "http://127.0.0.1:8090/v1"

# ---- temperature 策略（角色语义，非环境变量覆盖）----
PLANNER_TEMPERATURE = 0.3   # 知识模型构建：需一定创造性，但整体结构化
DIAGNOSER_TEMPERATURE = 0.0  # 诊断：锁死确定性（plan §1）
TEACHER_TEMPERATURE = 0.5   # 干预/探测：需自然的教学语言

# ---- 思考模式（DeepSeek V4 thinking）----
# deepseek-v4-flash / deepseek-v4-pro 均支持思考模式，通过顶层参数开启：
#   thinking={"type": "enabled"} + reasoning_effort="low|high|max"
# 思考链经响应里的 reasoning_content 字段返回（与最终 content 分离）。
# thinking 是 DeepSeek 非标准参数，OpenAI SDK 下须经 extra_body 传入；
# reasoning_effort 是标准顶层参数，直接用显式字段传递。
# 环境变量便于本地 adapter / 上游网关不支持时快速开关与调档。
THINKING_ENABLED = os.getenv("THINKING_ENABLED", "true").lower() not in ("0", "false", "no")
THINKING_EFFORT = os.getenv("THINKING_EFFORT", "high")  # low / high / max

def _thinking_kwargs() -> dict:
    """构造 ChatDeepSeek 的思考模式参数（thinking 开关 + 推理力度）。"""
    return {
        "reasoning_effort": THINKING_EFFORT,
        "extra_body": {
            "thinking": {"type": "enabled" if THINKING_ENABLED else "disabled"}
        },
    }


def _model_id(env_var: str, default: str) -> str:
    """从环境变量读取模型 ID，缺省用默认值。"""
    return os.getenv(env_var, default)


def _api_base() -> str:
    """LLM 接入地址（本地 OpenAI 兼容 adapter）。"""
    return os.getenv("LLM_API_BASE", DEFAULT_API_BASE)


def _api_key() -> str | None:
    """LLM 接入鉴权 key（adapter 的 ADAPTER_API_KEY），未配置则返回 None。"""
    return os.getenv("LLM_API_KEY") or None


def get_planner_model() -> ChatDeepSeek:
    """知识模型构建模型（planner_model）。"""
    return ChatDeepSeek(
        model=_model_id("PLANNER_MODEL", DEFAULT_PLANNER_MODEL),
        api_base=_api_base(),
        api_key=_api_key(),
        temperature=PLANNER_TEMPERATURE,
        timeout=120,
        max_retries=2,
        **_thinking_kwargs(),
    )


def get_teacher_model() -> ChatDeepSeek:
    """干预 / 探测生成模型（teacher_model）。"""
    return ChatDeepSeek(
        model=_model_id("TEACHER_MODEL", DEFAULT_TEACHER_MODEL),
        api_base=_api_base(),
        api_key=_api_key(),
        temperature=TEACHER_TEMPERATURE,
        timeout=120,
        max_retries=2,
        **_thinking_kwargs(),
    )


def get_diagnoser_model() -> ChatDeepSeek:
    """认知诊断模型（diagnoser_model），低 temperature 锁死确定性。"""
    return ChatDeepSeek(
        model=_model_id("DIAGNOSER_MODEL", DEFAULT_DIAGNOSER_MODEL),
        api_base=_api_base(),
        api_key=_api_key(),
        temperature=DIAGNOSER_TEMPERATURE,
        timeout=120,
        max_retries=2,
        **_thinking_kwargs(),
    )


def get_structured_diagnoser() -> Runnable:
    """返回绑定 Diagnosis schema 的结构化诊断模型。

    调用后直接返回 Diagnosis 实例（Pydantic），而非 AIMessage。
    用于 diagnose 节点的结构化输出（plan §4、任务④验收标准）。
    """
    return get_diagnoser_model().with_structured_output(Diagnosis)


def _extract_json(text: str) -> dict:
    """从模型输出中提取 JSON 对象（容错 markdown 代码块包裹）。"""
    t = text.strip()
    # 去掉 ```json ... ``` 包裹
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start:end + 1]
    return json.loads(t)


def invoke_structured(model, schema, messages):
    """结构化输出的降级兜底：普通 invoke + JSON Schema 强约束 + 手动解析。

    背景：本地 adapter（工蜂 Gateway）支持 function calling，但有两个不稳定点：
    1. 上游模型对模糊/非常规输入（如「开始」「你好」）可能**不调用工具**，直接
       `finish_reason=stop` 返回普通文本，导致 `with_structured_output` 返回 None
       （实测 `deepseek-v4-flash` 对「开始」返回 None、对「Spring AOP」正常）。
    2. 带嵌套对象引用（`$defs`/`$ref`）的复杂 schema 会让上游 fc 卡死（见
       graph._build_knowledge_model，该类 schema 走专门的普通文本方案）。

    因此对「扁平 schema + 可能模糊输入」的场景，当 `with_structured_output`
    返回 None 时降级到本函数：在 prompt 里注入 JSON Schema 强约束，让模型
    「只输出 JSON 本身」，再手动解析。实测 flash 对模糊输入走此路径稳定返回。
    """
    json_schema = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    instruction = (
        "请严格按以下 JSON Schema 输出一个 JSON 对象，不要输出任何解释、注释或 "
        "markdown 代码块，只输出 JSON 本身：\n"
        f"{json_schema}"
    )
    result = model.invoke([*messages, ("human", instruction)])
    content = result.content if hasattr(result, "content") else str(result)
    return schema.model_validate(_extract_json(content))


def structured_output(model, schema, messages):
    """结构化输出的统一入口：优先 with_structured_output，返回 None 则降级解析。

    背景见 invoke_structured：本地 adapter（工蜂 Gateway）对模糊/非常规输入可能
    不调用工具（finish_reason=stop），导致 with_structured_output 返回 None。此函数
    把「先尝试 function calling、失败再降级普通 invoke + 手动 JSON 解析」封装为单一
    入口，供 graph / learning_engine / conversation_agent 各处复用，避免重复实现。
    """
    result = model.with_structured_output(schema).invoke(messages)
    if result is not None:
        return result
    return invoke_structured(model, schema, messages)


def extract_reasoning_from_message(message) -> str:
    """从 LangChain 消息中提取思考链（reasoning_content）。

    ChatDeepSeek 会把 DeepSeek 的 reasoning_content 放进
    `message.additional_kwargs["reasoning_content"]`（流式为逐 delta，非流式为完整
    文本）。此函数统一提取，供 UI 展示「思考过程」。
    """
    if message is None:
        return ""
    additional = getattr(message, "additional_kwargs", None) or {}
    return additional.get("reasoning_content") or ""


def structured_output_with_reasoning(model, schema, messages):
    """结构化输出，并尽力提取思考内容（reasoning_content）。

    返回 `(parsed, reasoning_text)`。用于对话 Agent 场景：既要结构化决策结果，
    又要拿到思考链给用户看（通用 Agent 基座能力）。

    - 真实 ChatDeepSeek：`with_structured_output(include_raw=True)` 返回
      `{"raw": AIMessage, "parsed": ...}`，从 raw 提取 reasoning_content。
    - 测试桩：`with_structured_output` 只接受 `schema` 一个位置参数，传
      `include_raw=True` 会抛 TypeError，此时回退普通结构化输出、reasoning 为空，
      保证离线测试兼容。
    """
    try:
        bound = model.with_structured_output(schema, include_raw=True)
    except TypeError:
        # 测试桩（ScriptedLLM）不支持 include_raw：回退普通结构化输出
        return structured_output(model, schema, messages), ""

    raw_result = bound.invoke(messages)
    if isinstance(raw_result, dict):
        parsed = raw_result.get("parsed")
        raw = raw_result.get("raw")
        reasoning = extract_reasoning_from_message(raw)
        if parsed is not None:
            return parsed, reasoning
        # parsed 为 None（模型未调用工具）→ 走降级解析
        return structured_output(model, schema, messages), ""
    # 非 dict（如测试桩直接返回 Pydantic 对象）
    return raw_result, ""


def get_conversation_agent_model() -> ChatDeepSeek:
    """对话 Agent 模型（理解用户意图 + 决定教学动作 + 生成自然回复）。

    对话 Agent 是新增的独立角色（与 Learning Engine 解耦），但底层仍复用 teacher 的
    temperature 策略（自然教学语言）。单独命名以明确语义，后续可独立调参或替换。
    """
    return get_teacher_model()
