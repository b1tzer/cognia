"""学习者画像：跨会话长期记忆核心逻辑（Phase 3）。

对应 TASA 遗忘曲线 + Chudziak & Kostka (2025) 的 LTM prior knowledge：
- apply_forgetting：Ebbinghaus 遗忘衰减
- persist_concept_mastery：目标完成时沉淀掌握度
- prior_mastery：新目标用历史掌握度做先验
"""
from __future__ import annotations

from datetime import datetime, timezone

import db

# 遗忘半衰期（天）：掌握度每过 half_life 天衰减为原来的一半（Ebbinghaus 曲线）
FORGETTING_HALF_LIFE_DAYS = 30.0
# 衰减下限：长时间不复习，掌握度最低回落到这里（不衰减到 0，保留一点印象）
FORGETTING_FLOOR = 0.05


def _elapsed_days(ts: str) -> float:
    """从 ISO 时间戳计算距今多少天（负值按 0 处理，解析失败按 0 处理）。"""
    try:
        t = datetime.fromisoformat(ts)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        delta = (datetime.now(timezone.utc) - t).total_seconds() / 86400.0
        return max(0.0, delta)
    except Exception:
        return 0.0


def apply_forgetting(mastery: float, elapsed_days: float) -> float:
    """遗忘衰减：retention = 2^(-t/half_life)，设下限 FORGETTING_FLOOR。

    - elapsed_days <= 0 时不衰减
    - 每过 half_life 天保留率减半，时间越长衰减越多，最终趋近但不低于 FORGETTING_FLOOR
    """
    if elapsed_days <= 0:
        return max(0.0, min(1.0, mastery))
    retention = 0.5 ** (elapsed_days / FORGETTING_HALF_LIFE_DAYS)
    decayed = mastery * retention
    return max(FORGETTING_FLOOR, min(1.0, decayed))


def persist_concept_mastery(cognitive: dict, user_id: str = db.DEFAULT_USER_ID) -> None:
    """目标完成时，把 cognitive 各概念的 mastery 沉淀到 concept_mastery。"""
    for m in cognitive.get("concepts", []):
        cid = m.get("concept_id")
        if not cid:
            continue
        db.upsert_concept_mastery(
            user_id,
            cid,
            float(m.get("mastery", 0.0)),
            m.get("last_evidence", ""),
        )


def prior_mastery(user_id: str = db.DEFAULT_USER_ID) -> dict:
    """返回 {concept_id: 遗忘衰减后的 mastery}，用于新目标先验。"""
    ledger = db.get_concept_mastery(user_id)
    out = {}
    for cid, row in ledger.items():
        days = _elapsed_days(row.get("updated_at", ""))
        out[cid] = apply_forgetting(float(row.get("mastery", 0.0)), days)
    return out
