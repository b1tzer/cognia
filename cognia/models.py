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
- .env 的加载（load_dotenv）由应用入口负责，本模块保持纯净、只从 os.getenv
  读取，便于测试与复用。
- 「独立严格 prompt」不在此层实现：诊断 system prompt 需引用知识模型、当前
  知识点与运行时上下文，属诊断逻辑的职责。本层只负责模型实例化
  （model / temperature / timeout）与结构化输出绑定。
"""

import json
import os

from langchain_deepseek import ChatDeepSeek

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

# ---- 思考模式（工蜂 Gateway external 模型默认返回 reasoning_content）----
# 实测（.cache/debug_thinking_matrix.py）：工蜂 Gateway 的 external 模型
# （deepseek-v4-flash-external / deepseek-v4-pro）不传任何 thinking 参数时，
# 默认就在响应的 reasoning_content 字段返回完整思考链；而显式传
# thinking={"type":"enabled"} 反而会抑制 reasoning_content（pro 直接变 None，
# flash-external 变超短）。internal 版 deepseek-v4-flash 则不返回思考链。
# 因此本函数返回空 dict：不额外传 thinking / reasoning_effort，让 external
# 模型的默认行为生效。THINKING_ENABLED / THINKING_EFFORT 环境变量已废弃，
# 仅保留定义以兼容旧 .env，不再被读取。
THINKING_ENABLED = os.getenv("THINKING_ENABLED", "true").lower() not in ("0", "false", "no")
THINKING_EFFORT = os.getenv("THINKING_EFFORT", "high")  # low / high / max

def _thinking_kwargs() -> dict:
    """返回空参数（不显式传 thinking），让 external 模型默认返回 reasoning_content。"""
    return {}


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


def _strip_trailing_commas(t: str) -> str:
    """移除 JSON 结构中的尾逗号（`[...,]` / `{...,}`），字符串内的逗号不受影响。

    用逐字符扫描跟踪是否处于字符串内，只跳过「非字符串内、且下一个非空白字符
    是 `}` 或 `]`」的逗号，避免误删字符串内容里的逗号。
    """
    out = []
    in_string = False
    escape = False
    i = 0
    n = len(t)
    while i < n:
        ch = t[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
            out.append(ch)
        elif ch == ",":
            j = i + 1
            while j < n and t[j] in " \t\r\n":
                j += 1
            if j < n and t[j] in "}]":
                # 尾逗号：跳过不输出
                i += 1
                continue
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _insert_missing_commas(t: str) -> str:
    """补齐 JSON 中「相邻值之间漏写逗号」的错误（对应 "Expecting ',' delimiter"）。

    同样跟踪字符串状态，仅在非字符串内发现闭合括号 `}` / `]` 后紧跟新值开头
    （`{` / `[` / `"`）时插入逗号，避免破坏字符串内容。
    """
    out = []
    in_string = False
    escape = False
    i = 0
    n = len(t)
    while i < n:
        ch = t[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        if ch in "}]":
            j = i + 1
            while j < n and t[j] in " \t\r\n":
                j += 1
            if j < n and t[j] in '{"[':
                out.append(",")
        i += 1
    return "".join(out)


def _repair_json(t: str) -> str:
    """对 LLM 输出 JSON 做低风险自动修复（尾逗号 / 缺失逗号），返回修复文本。"""
    return _insert_missing_commas(_strip_trailing_commas(t))


def _extract_json(text: str):
    """从模型输出中提取 JSON（容错 markdown 代码块包裹），返回解析后的对象。

    优先按数组 `[...]` 提取（知识建模场景输出扁平 JSON 数组），否则回退到
    对象 `{...}`（结构化输出场景）。供本模块的 invoke_structured 与
    learning_engine.build_knowledge_model 复用，避免重复实现。

    解析失败时先做一次低风险自动修复（尾逗号 / 缺失逗号），仍失败则抛出
    JSONDecodeError，交由调用方决定是否重试（见 learning_engine.build_knowledge_model）。
    """
    t = text.strip()
    # 去掉 ```json ... ``` 包裹
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    start = t.find("[")
    end = t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        start = t.find("{")
        end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        t = t[start:end + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return json.loads(_repair_json(t))


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
    """结构化输出的统一入口：function calling（tool_choice=auto），失败/None 降级解析。

    实测定位（2026-09-22，见 scripts/bisect_502.py）：本地 adapter（工蜂 Gateway）
    不支持 function calling 的「强制指定工具名」tool_choice（即
    {"type":"function","function":{"name":...}}），会以 502/400
    "rejected by an internal MaaS component" 拒绝请求。而 LangChain 的
    with_structured_output(method="function_calling") 默认就是强制指定工具名
    （tool_choice=tool_name），这正是此前诊断崩溃（SocketError "other side closed" /
    INCOMPLETE_STREAM）的根因——不是 schema 嵌套引用问题，也不是模型不可控。

    修复：显式传 tool_choice="auto" 覆盖 LangChain 默认，让模型自行决定是否调用工具，
    实测 deepseek-v4-pro 走 auto 稳定返回结构化结果（Diagnosis）。

    仍保留降级：若 function calling 返回 None（模糊输入，如「开始」）或抛异常，
    降级到普通 invoke + JSON Schema 强约束 + 手动解析，避免炸穿 tool 节点中断 SSE 流。
    """
    try:
        result = model.with_structured_output(schema, tool_choice="auto").invoke(messages)
        if result is not None:
            return result
    except Exception as exc:  # 网关异常等，降级而非崩溃
        print(f"[Cognia] with_structured_output 失败，降级到普通 JSON 解析：{exc}")
    return invoke_structured(model, schema, messages)


def get_conversation_agent_model() -> ChatDeepSeek:
    """对话 Agent 模型（理解用户意图 + 决定教学动作 + 生成自然回复）。

    对话 Agent 是新增的独立角色（与 Learning Engine 解耦），但底层仍复用 teacher 的
    temperature 策略（自然教学语言）。单独命名以明确语义，后续可独立调参或替换。
    """
    return get_teacher_model()
