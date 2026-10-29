#!/usr/bin/env python3
"""临时取证脚本：检查最新会话 checkpoint_blobs 里 messages 的结构。

LangGraph Postgres checkpointer 把 messages 存进 checkpoint_blobs 表
(channel='messages', type='msgpack')，而非内联在 checkpoint JSONB 的
channel_values。本脚本直接读该 blob 并反序列化。

目的：定位「知识模型 JSON 气泡」到底来自
  (A) 模型在 assistant 最终回复里真复述的 JSON（content 为 JSON 字符串），还是
  (B) 只存在于 tool 结果消息（role=tool / 带 tool_call_id），由前端运行时当成气泡渲染。

用法:
  .venv/bin/python scripts/inspect_session.py               # 最新 1 个会话
  .venv/bin/python scripts/inspect_session.py --threads 4  # 最新 4 个会话
"""
import sys
import json
import argparse
import msgpack

try:
    import psycopg
except ImportError:
    print("psycopg 未安装，请使用项目 .venv")
    sys.exit(1)

URL = "postgresql://postgres:password@127.0.0.1:5432/cognia"


def is_json_blob(s: str) -> bool:
    t = s.strip()
    if not (t.startswith("[") or t.startswith("{")):
        return False
    try:
        json.loads(t)
        return True
    except Exception:
        return False


def ext_hook(code, data):
    # LangGraph 用 msgpack ext 序列化 BaseMessage：ext code=5 的 data 是
    # [module, classname, dict, 'model_validate_json']，真实字段在第 3 个 dict 里。
    if code == 5:
        try:
            inner = msgpack.unpackb(data, raw=False)
            if isinstance(inner, list) and len(inner) >= 3 and isinstance(inner[2], dict):
                return inner[2]
        except Exception:
            pass
    try:
        text = data.decode("utf-8", "replace")
    except Exception:
        text = repr(data)
    return {"__ext__": code, "data": text}


def str_of(obj, limit=200):
    if isinstance(obj, str):
        return obj[:limit].replace("\n", "\\n")
    if isinstance(obj, (dict, list)):
        try:
            return json.dumps(obj, ensure_ascii=False)[:limit]
        except Exception:
            return str(obj)[:limit]
    return str(obj)[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    conn = psycopg.connect(URL, connect_timeout=10)
    cur = conn.cursor()

    # 取每个 thread 的所有 messages blob，挑 version 最大的（最新状态）
    cur.execute("""
        SELECT thread_id, version, blob
        FROM checkpoint_blobs
        WHERE channel = 'messages' AND checkpoint_ns = ''
    """)
    by_thread = {}
    for tid, ver, blob in cur.fetchall():
        if tid not in by_thread or ver > by_thread[tid][0]:
            by_thread[tid] = (ver, blob)

    # 按 thread_id 字符串倒序近似"最新会话"（thread_id 含时间戳时更准确；
    # 这里简单取全部，再截最新 N 个）
    threads = sorted(by_thread.items(), key=lambda kv: kv[0], reverse=True)[: args.threads]
    print(f"共 {len(by_thread)} 个会话含 messages，本次检查最新 {len(threads)} 个\n")

    for tid, (ver, blob) in threads:
        print("=" * 70)
        print(f"THREAD: {tid}  (messages version={ver})")
        try:
            msgs = msgpack.unpackb(blob, ext_hook=ext_hook, raw=False)
        except Exception as e:
            print("  反序列化失败:", e)
            continue
        if not isinstance(msgs, list):
            print("  messages 非 list:", type(msgs))
            msgs = []
        print(f"消息条数: {len(msgs)}\n")
        for idx, m in enumerate(msgs):
            if isinstance(m, dict):
                # LangGraph BaseMessage 解包后是 dict，字段用 type 而非 role
                role = m.get("type") or m.get("role")
                mid = m.get("id") or ""
                name = m.get("name") or ""
                tcid = m.get("tool_call_id") or ""
                content = m.get("content")
                tool_calls = m.get("tool_calls") or m.get("tool_calls_") or []
                tcnames = [tc.get("name") for tc in tool_calls if isinstance(tc, dict)]
            elif isinstance(m, str):
                role, mid, name, tcid, content, tcnames = "str", "", "", "", m, []
            else:
                role, mid, name, tcid, content, tcnames = type(m).__name__, "", "", "", str(m), []
            if isinstance(content, str):
                ctype = f"str(len={len(content)})"
                flagged = " <<< JSON!" if is_json_blob(content) else ""
            else:
                ctype = type(content).__name__
                flagged = ""
            print(f"[{idx:02d}] type={str(role):<10} id={str(mid)[:8]:<8} name={name:<16} "
                  f"tool_call_id={str(tcid)[:8]:<8} content={ctype} tool_calls={tcnames}{flagged}")
            if flagged:
                print(f"      JSON content 预览: {str_of(content, 400)}")
        print()
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()