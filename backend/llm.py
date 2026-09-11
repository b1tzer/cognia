"""LLM 客户端封装（OpenAI 兼容协议）。

默认对接本机 TencentDB Agent Memory 使用的 adapter.py（127.0.0.1:8090）。
该代理背后是带 reasoning 的推理模型：content 常被 ```json 包裹，且
reasoning_content 会消耗大量 token。因此这里做了健壮的 JSON 提取、调大了
max_tokens，并在 content 为空时回退到 reasoning_content。

AI_ENABLED=False 时回退到启发式/模板引擎（见 domain_model.py 与 tutor.py），
产品在无外部模型的情况下依然可完整运行闭环。
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

import config

# ---------------------------------------------------------------------------
# token 用量观测（用于成本核算，落地「节约」约束）
# ---------------------------------------------------------------------------
_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

def reset_usage() -> None:
    """清零 token 用量计数器。"""
    for k in _usage:
        _usage[k] = 0

def get_usage() -> dict[str, int]:
    """返回累计 token 用量快照（副本，避免外部误改）。"""
    return dict(_usage)

def _record_usage(resp: Any) -> None:
    """从 OpenAI 响应累加 token 用量（含重试失败轮次的真实消耗）。"""
    usage = getattr(resp, "usage", None)
    if not usage:
        return
    _usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
    _usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
    _usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
    _usage["calls"] += 1


def _record_trace(
    trace: Optional[list],
    label: str,
    model: str,
    system: str,
    user: str,
    output: str,
    usage: Any,
) -> None:
    """把一次 LLM 调用记录进思考轨迹（trace）。

    trace 为调用方传入的 list（可跨多层累积），每条记录保存：
    - label：层中文标签（如「认知诊断」）
    - model：实际使用的模型
    - system / user：发给 LLM 的完整 prompt
    - output：LLM 原始输出（未解析）
    - usage：本次 token 用量

    trace 为 None 时跳过（不采集），保证不影响正常流程。
    """
    if trace is None:
        return
    u = usage or {}
    trace.append({
        "label": label,
        "model": model,
        "system": system,
        "user": user,
        "output": output or "",
        "usage": {
            "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(u, "total_tokens", 0) or 0),
        },
    })

def _extract_json(text: str) -> Any:
    """从模型输出中稳健地提取 JSON 对象/数组。

    兼容推理模型的 ```json ... ``` 包裹、前后缀说明文字等噪声。
    """
    text = (text or "").strip()
    if not text:
        return None
    # 1) 剥离 ```json / ``` 代码块围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # 2) 截取首个 { 或 [ 到最后一个 } 或 ]
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1),
        default=-1,
    )
    if start == -1:
        return None
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _model_candidates() -> list[str]:
    """按顺序返回要尝试的模型：主模型 + 备选模型（去重）。"""
    models = [config.OPENAI_MODEL]
    fallback = getattr(config, "OPENAI_MODEL_FALLBACK", "")
    if fallback and fallback != config.OPENAI_MODEL:
        models.append(fallback)
    return models


def chat_json(
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    trace: Optional[list] = None,
    trace_label: str = "LLM 调用",
) -> Optional[dict]:
    """调用 LLM 并返回 JSON 对象。失败或未启用时返回 None，由调用方降级。

    trace 传入时，会把本次调用的 prompt 与原始输出记录进去（用于「思考过程」展示）。
    """
    if not config.AI_ENABLED:
        return None
    try:
        from openai import OpenAI

        # 直通 fast 模型（deepseek-v4-flash 等）响应很快，300s 超时足够；
        # 失败（429/5xx/网络）则自动切换到备选 fast 模型。
        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=300.0,
        )
        for model in _model_candidates():
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                _record_usage(resp)
                raw = resp.choices[0].message.content or ""
                _record_trace(trace, trace_label, model, system, user, raw, getattr(resp, "usage", None))
                data = _extract_json(raw)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
        return None
    except Exception:
        return None


def chat_text(
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    trace: Optional[list] = None,
    trace_label: str = "LLM 调用",
) -> Optional[str]:
    """调用 LLM 返回纯文本。失败时返回 None。"""
    if not config.AI_ENABLED:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=300.0,
        )
        for model in _model_candidates():
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                _record_usage(resp)
                raw = resp.choices[0].message.content or ""
                _record_trace(trace, trace_label, model, system, user, raw, getattr(resp, "usage", None))
                content = raw.strip()
                # content 为空时回退到 reasoning_content（兼容极少数推理模型）
                if not content:
                    rc = getattr(resp.choices[0].message, "reasoning_content", None)
                    if rc:
                        content = rc.strip()
                if content:
                    return content
            except Exception:
                continue
        return None
    except Exception:
        return None


def chat_text_stream(
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    trace: Optional[list] = None,
    trace_label: str = "LLM 调用",
):
    """流式调用 LLM，逐个 yield 文本增量（delta）。失败时 yield 空（无输出）。

    返回生成器，调用方用 `for delta in chat_text_stream(...)` 消费。
    流式响应最后一个 chunk 携带 usage，会自动累加到 token 用量观测。
    未启用 AI 或全程无有效输出时，该生成器不会 yield 任何内容（由调用方降级）。
    """
    if not config.AI_ENABLED:
        return
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=300.0,
        )
        for model in _model_candidates():
            try:
                stream = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                emitted = False
                parts: list[str] = []
                usage = None
                for chunk in stream:
                    # 流式响应最后一个 chunk 携带 usage（配合 include_usage）
                    if getattr(chunk, "usage", None):
                        _record_usage(chunk)
                        usage = getattr(chunk, "usage", None)
                    choices = getattr(chunk, "choices", None)
                    if (
                        choices
                        and choices[0].delta
                        and getattr(choices[0].delta, "content", None)
                    ):
                        emitted = True
                        part = choices[0].delta.content
                        parts.append(part)
                        yield part
                if emitted:
                    _record_trace(trace, trace_label, model, system, user, "".join(parts), usage)
                    return
            except Exception:
                continue
    except Exception:
        return
