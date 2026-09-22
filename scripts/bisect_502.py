#!/usr/bin/env python3
"""对照定位触发 502 的参数。

LangChain with_structured_output 会强制加 tool_choice=工具名 + parallel_tool_calls=False，
而手工 tool_choice=auto 能成功。逐个对照，定位触发网关 502 的字段。
"""

import json
import os
import time

import httpx

BASE = os.getenv("LLM_API_BASE", "http://127.0.0.1:8090/v1")
KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("DIAGNOSER_MODEL", "deepseek-v4-pro")

FLAT_TOOL = {
    "type": "function",
    "function": {
        "name": "Diagnosis",
        "description": "提交一次认知诊断观察",
        "parameters": {
            "type": "object",
            "properties": {
                "point_id": {"type": "string"},
                "state": {"type": "string", "enum": ["unassessed", "mastered", "partial", "misconception", "unknown"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "evidence": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["point_id", "state", "confidence"],
        },
    },
}


def call(label, body):
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    body = dict(body, model=MODEL)
    t0 = time.time()
    try:
        r = httpx.post(f"{BASE}/chat/completions", json=body, headers=headers, timeout=120)
        dt = time.time() - t0
        txt = r.text[:200].replace("\n", " ")
        print(f"[{label}] HTTP {r.status_code} ({dt:.1f}s): {txt}")
        return r.status_code
    except Exception as e:  # noqa: BLE001
        print(f"[{label}] EXC {type(e).__name__}: {e}")
        return None


base = {
    "messages": [{"role": "user", "content": "诊断：用户说 AOP 就是切面"}],
    "temperature": 0.0,
    "stream": False,
    "tools": [FLAT_TOOL],
}

print("=" * 70)
call("A. tool_choice=auto（手工基线，预期 200）", {**base, "tool_choice": "auto"})

print("-" * 70)
call("B. tool_choice=强制指定函数名（LangChain 行为）",
     {**base, "tool_choice": {"type": "function", "function": {"name": "Diagnosis"}}})

print("-" * 70)
call("C. tool_choice=auto + parallel_tool_calls=false",
     {**base, "tool_choice": "auto", "parallel_tool_calls": False})

print("-" * 70)
call("D. 完整复刻 LangChain（强制名 + parallel_tool_calls=false）",
     {**base, "tool_choice": {"type": "function", "function": {"name": "Diagnosis"}}, "parallel_tool_calls": False})

print("-" * 70)
call("E. 强制名 + strict=true",
     {**base, "tool_choice": {"type": "function", "function": {"name": "Diagnosis"}}, "strict": True})