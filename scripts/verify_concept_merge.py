#!/usr/bin/env python3
"""实测知识点 name 的语义相似度，验证「概念语义合并」根因。

背景：知识模型把「happens-before 原则定义 / 具体规则 / 原则的应用」三个点
合并成了同一个 point_id（concept:b8721b67）。本脚本用 embedding 计算它们两两
cosine，确认是否因语义相似度 ≥ 阈值（默认 0.85）被误合并。

用法：
    uv run python scripts/verify_concept_merge.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognia import embedding

NAMES = [
    "happens-before 原则定义",
    "happens-before 具体规则",
    "happens-before 原则的应用",
    "Java 内存模型(JMM)基本概念",
    "内存可见性问题",
    "原子性在 JMM 中的含义",
]


def main() -> int:
    model = embedding.get_embedding_model()
    if model is None:
        print("embedding 模型不可用（未安装 fastembed / 模型未下载）")
        print("若如此，合并应退化为 name 精确匹配，三个 name 不同则不会合并")
        return 1

    vecs: dict[str, list[float] | None] = {}
    for n in NAMES:
        vecs[n] = embedding.embed_text(n)
        print(f"{n}: embedding={'ok' if vecs[n] else 'None'}")

    print("\n--- 两两 cosine（>=0.85 会被 merge_concept 合并）---")
    for i in range(len(NAMES)):
        for j in range(i + 1, len(NAMES)):
            a, b = NAMES[i], NAMES[j]
            s = embedding.cosine(vecs[a], vecs[b])
            flag = "  <<< 会合并" if s >= 0.85 else ""
            print(f"{a}\n    vs {b}\n    => {s:.4f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
