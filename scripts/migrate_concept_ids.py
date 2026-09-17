#!/usr/bin/env python3
"""Cognia 概念 ID 迁移脚本：把存量知识点 point_id 重写为跨对话全局稳定 id。

背景：知识版图从「按对话（goal）隔离」升级为「按用户全局聚合」。旧数据里
point_id 是 LLM 裸生成的、仅单 goal 内唯一；本脚本对存量知识点做语义合并，
把 point_id 重写为全局稳定 id（concept:{sha1(归一化name)[:8]}），并同步重写
observation / proficiency 里的 point_id 与 key，保证熟练度跨对话正确聚合。

用法示例：
    # 预览某 user 的迁移映射（不写库，安全）
    uv run python scripts/migrate_concept_ids.py --user-id <uuid> --dry-run

    # 实际执行迁移
    uv run python scripts/migrate_concept_ids.py --user-id <uuid>

    # 迁移所有 user（前缀扫描 namespace）
    uv run python scripts/migrate_concept_ids.py --all --dry-run

    # 覆盖 embedding 模型（默认读 .env 的 EMBEDDING_MODEL）
    uv run python scripts/migrate_concept_ids.py --user-id <uuid> --embedding-model BAAI/bge-small-zh-v1.5

依赖：项目已有依赖（langgraph / fastembed / dotenv），无需额外安装。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# 项目根目录加入 sys.path，保证脚本从任意 cwd 独立运行
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from cognia import concept_merge, embedding
from cognia.memory import (
    CONCEPT_NS,
    KNOWLEDGE_MODEL_NS,
    OBSERVATION_NS,
    PROFICIENCY_NS,
    get_store,
)


def _key_of(value: dict, timestamp_key: str = "timestamp") -> str | None:
    """从 value 里取 timestamp（observation / proficiency 的 key 组成）。"""
    ts = value.get(timestamp_key)
    return ts if isinstance(ts, str) else None


def _split_point_key(key: str) -> tuple[str, str] | None:
    """把 `point_id:timestamp` 拆成 (point_id, timestamp)。"""
    if ":" not in key:
        return None
    head, tail = key.split(":", 1)
    return head, tail


async def _migrate_user(store, user_id: str, dry_run: bool) -> dict:
    """迁移单个 user：重写 knowledge_model / observation / proficiency 的 point_id。"""
    report = {
        "user_id": user_id,
        "goals": 0,
        "points_migrated": 0,
        "observations_migrated": 0,
        "proficiencies_migrated": 0,
        "id_map": {},
    }

    # 1) 读该 user 现有全局概念（含 embedding）
    concept_items = await store.asearch((CONCEPT_NS, user_id))
    concepts = [it.value for it in concept_items if it.value]

    # 2) 读全部知识模型，逐点语义合并，建立 旧id → 新id 映射
    km_items = await store.asearch((KNOWLEDGE_MODEL_NS, user_id))
    for it in km_items:
        km = it.value
        if not km:
            continue
        report["goals"] += 1
        points = km.get("points") or []
        id_map: dict[str, str] = {}
        for p in points:
            old_id = p.get("id")
            name = p.get("name") or old_id
            global_id, is_new, canonical = concept_merge.merge_concept(name, concepts)
            id_map[old_id] = global_id
            report["id_map"][old_id] = global_id
            if is_new:
                vec = embedding.embed_text(canonical)
                concepts.append({"id": global_id, "name": canonical, "embedding": vec})
                if not dry_run:
                    await store.aput(
                        (CONCEPT_NS, user_id),
                        global_id,
                        {"id": global_id, "name": canonical, "embedding": vec or []},
                    )
            p["id"] = global_id
            report["points_migrated"] += 1
        # 重写 prerequisites 里的裸 id 引用
        for p in points:
            p["prerequisites"] = [
                id_map.get(pid, pid) for pid in (p.get("prerequisites") or [])
            ]
        if not dry_run:
            await store.aput((KNOWLEDGE_MODEL_NS, user_id), it.key, km)

    # 3) 重写 observation 的 point_id（删旧 key 写新 key）
    obs_items = await store.asearch((OBSERVATION_NS, user_id))
    for it in obs_items:
        value = it.value
        if not value:
            continue
        old_id = value.get("point_id")
        new_id = report["id_map"].get(old_id)
        if not new_id or new_id == old_id:
            continue
        ts = _key_of(value)
        if not ts:
            continue
        value["point_id"] = new_id
        new_key = f"{new_id}:{ts}"
        if not dry_run:
            await store.aput((OBSERVATION_NS, user_id), new_key, value)
            await store.adelete((OBSERVATION_NS, user_id), it.key)
        report["observations_migrated"] += 1

    # 4) 重写 proficiency（旧 Delta）的 point_id
    prof_items = await store.asearch((PROFICIENCY_NS, user_id))
    for it in prof_items:
        value = it.value
        if not value:
            continue
        old_id = value.get("point_id")
        new_id = report["id_map"].get(old_id)
        if not new_id or new_id == old_id:
            continue
        ts = _key_of(value)
        if not ts:
            continue
        value["point_id"] = new_id
        new_key = f"{new_id}:{ts}"
        if not dry_run:
            await store.aput((PROFICIENCY_NS, user_id), new_key, value)
            await store.adelete((PROFICIENCY_NS, user_id), it.key)
        report["proficiencies_migrated"] += 1

    return report


async def _migrate_all(store, dry_run: bool) -> list[dict]:
    """前缀扫描全部 user（namespace 第一段 = KNOWLEDGE_MODEL_NS）。"""
    items = await store.asearch((KNOWLEDGE_MODEL_NS,))
    user_ids: set[str] = set()
    for it in items:
        ns = it.namespace
        if len(ns) >= 2 and ns[1]:
            user_ids.add(ns[1])
    reports = []
    for uid in sorted(user_ids):
        reports.append(await _migrate_user(store, uid, dry_run))
    return reports


async def _main(args) -> None:
    if args.embedding_model:
        os.environ["EMBEDDING_MODEL"] = args.embedding_model

    store = await get_store()

    if args.all:
        reports = await _migrate_all(store, args.dry_run)
    elif args.user_id:
        reports = [await _migrate_user(store, args.user_id, args.dry_run)]
    else:
        raise SystemExit("必须指定 --user-id 或 --all")

    print(json.dumps(
        {"dry_run": args.dry_run, "reports": reports},
        ensure_ascii=False,
        indent=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把存量知识点 point_id 重写为跨对话全局稳定 id（语义合并）"
    )
    parser.add_argument("--user-id", default=None, help="指定要迁移的 user_id")
    parser.add_argument("--all", action="store_true", help="迁移所有 user（前缀扫描）")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="预览映射不写库（默认开启，安全）",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="可选：覆盖 embedding 模型（默认读 .env 的 EMBEDDING_MODEL）",
    )
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
