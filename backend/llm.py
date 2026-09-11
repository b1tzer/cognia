"""LLM 客户端封装（OpenAI 兼容协议）。

当未配置 API Key 时，AI_ENABLED 为 False，
上层调用会回退到启发式 / 模板引擎（见 domain_model.py 与 tutor.py），
保证产品在无外部模型的情况下依然可完整运行闭环。
"""
from __future__ import annotations

import json
from typing import Any, Optional

import config


def chat_json(
    system: str,
    user: str,
    temperature: float = 0.4,
    max_tokens: int = 4096,
) -> Optional[dict]:
    """调用 LLM 并强制返回 JSON 对象。

    失败或未启用时返回 None，由调用方降级。
    """
    if not config.AI_ENABLED:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
        text = text.strip()
        # 去除可能的代码块包裹
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception:
        return None


def chat_text(
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Optional[str]:
    """调用 LLM 返回纯文本。失败时返回 None。"""
    if not config.AI_ENABLED:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content
    except Exception:
        return None
