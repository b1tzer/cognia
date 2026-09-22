#!/usr/bin/env python3
"""诊断 cognia 模型调用情况，对比不同 function calling 格式是否被网关接受。

背景：DIAGNOSER_MODEL=deepseek-v4-pro 走 with_structured_output（function calling）
时被上游网关以 502/400 "rejected by an internal MaaS component" 拒绝，怀疑是
Pydantic model_json_schema 生成的 $defs/$ref 嵌套引用不被网关 MaaS 接受。

本脚本对比三种请求：
1. 普通 chat（无 tools）—— 基线，确认模型本身可用。
2. function calling（扁平 schema，enum 内联，模拟 hermes/Anthropic 风格）。
3. function calling（Pydantic $defs/$ref 嵌套，复现 cognia 实际发送的格式）。

用法：
    python3 scripts/test_model_call.py [--model deepseek-v4-pro] [--base http://127.0.0.1:8090/v1] [--key <adapter-key>]

可选：--direct 直连工蜂网关（模拟 hermes 路径，需 GF_* 环境变量）。
"""

import argparse
import json
import os
import sys
import time

import httpx


def _parse_args():
    p = argparse.ArgumentParser(description="诊断 cognia 模型调用")
    p.add_argument("--model", default="deepseek-v4-pro")
    p.add_argument("--base", default="http://127.0.0.1:8090/v1")
    p.add_argument("--key", default=os.getenv("LLM_API_KEY", ""))
    p.add_argument("--direct", action="store_true", help="直连工蜂网关（模拟 hermes）")
    return p.parse_args()


def _chat(base: str, key: str, model: str, body: dict, direct: bool):
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if direct:
        # 直连网关：使用工蜂三 header（同 hermes adapter 逻辑）
        headers = {
            "X-Username": os.getenv("GF_USERNAME", ""),
            "OAUTH-TOKEN": os.getenv("GF_TOKEN", ""),
            "DEVICE-ID": os.getenv("MOLTBOT_CLIENT_UUID", "") or os.getenv("GF_DEVICE_ID", ""),
            "Content-Type": "application/json",
            "X-Model-Name": model,
        }
    body = dict(body, model=model)
    t0 = time.time()
    try:
        r = httpx.post(url, json=body, headers=headers, timeout=90)
        dt = time.time() - t0
        print(f"    HTTP {r.status_code}  耗时 {dt:.1f}s")
        text = r.text
        if len(text) > 1500:
            text = text[:1500] + f"\n    ... (截断，共 {len(r.text)} 字符)"
        print(f"    {text}")
        return r.status_code, r.text
    except Exception as e:  # noqa: BLE001
        print(f"    请求异常: {type(e).__name__}: {e}")
        return None, str(e)


# ---- 扁平 schema（enum 内联，无 $defs/$ref，模拟 hermes 风格）----
FLAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "propose_diagnosis",
            "description": "提交一次认知诊断观察",
            "parameters": {
                "type": "object",
                "properties": {
                    "point_id": {"type": "string"},
                    "state": {"type": "string", "enum": ["unassessed", "mastered", "partial", "misconception", "unknown"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["point_id", "state", "confidence", "evidence"],
            },
        },
    }
]

# ---- Pydantic 嵌套 schema（含 $defs/$ref，模拟 cognia 实际发送）----
NESTED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "propose_diagnosis",
            "description": "提交一次认知诊断观察",
            "parameters": {
                "$defs": {
                    "CognitiveState": {"enum": ["unassessed", "mastered", "partial", "misconception", "unknown"], "title": "CognitiveState", "type": "string"},
                    "Confidence": {"enum": ["high", "medium", "low"], "title": "Confidence", "type": "string"},
                },
                "properties": {
                    "point_id": {"title": "Point Id", "type": "string"},
                    "state": {"$ref": "#/$defs/CognitiveState"},
                    "confidence": {"$ref": "#/$defs/Confidence"},
                    "evidence": {"items": {"type": "string"}, "title": "Evidence", "type": "array"},
                },
                "required": ["point_id", "state", "confidence", "evidence"],
                "title": "Diagnosis",
                "type": "object",
            },
        },
    }
]


def main():
    args = _parse_args()
    print(f"目标: model={args.model!r} base={args.base!r} direct={args.direct}")
    print("=" * 70)

    print("\n[1] 普通 chat（无 tools）—— 基线")
    _chat(args.base, args.key, args.model, {
        "messages": [{"role": "user", "content": "只回复两个字：你好"}],
        "temperature": 0.0,
        "stream": False,
    }, args.direct)

    print("\n[2] function calling（扁平 schema，enum 内联）")
    _chat(args.base, args.key, args.model, {
        "messages": [{"role": "user", "content": "诊断：用户说 AOP 就是切面"}],
        "temperature": 0.0,
        "stream": False,
        "tools": FLAT_TOOLS,
        "tool_choice": "auto",
    }, args.direct)

    print("\n[3] function calling（Pydantic $defs/$ref 嵌套）—— 复现 502")
    _chat(args.base, args.key, args.model, {
        "messages": [{"role": "user", "content": "诊断：用户说 AOP 就是切面"}],
        "temperature": 0.0,
        "stream": False,
        "tools": NESTED_TOOLS,
        "tool_choice": "auto",
    }, args.direct)


if __name__ == "__main__":
    main()
