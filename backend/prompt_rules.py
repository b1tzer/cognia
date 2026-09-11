"""各层 Prompt 动态规则的统一存取与注入。

「在线 Prompt 优化反馈回路」的持久化载体：每一层（诊断/决策/回复/知识模型）
都有独立的规则文件，optimizer 定期从真实对话中提炼规则写入，各层在构建
system prompt 时通过 rules_suffix() 追加这些规则，实现「随对话自我迭代」。

设计：
- 规则文件为 JSON，结构 {"version":1, "description":"...", "rules":["..."]}
- 诊断层复用既有 eval/diagnosis_rules.json（兼容 iterate.py 与认知层）
- 其余三层各建独立文件，避免相互污染
"""
from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent / "eval"

# 层 -> 规则文件路径。diagnosis 复用既有文件，其余为新增。
LAYER_RULES_PATH = {
    "diagnosis": EVAL_DIR / "diagnosis_rules.json",
    "decision_action": EVAL_DIR / "decision_rules.json",
    "tutor": EVAL_DIR / "tutor_rules.json",
    "domain_model": EVAL_DIR / "domain_model_rules.json",
}

LAYER_LABELS = {
    "diagnosis": "诊断层",
    "decision_action": "教学决策层",
    "tutor": "回复层",
    "domain_model": "知识模型层",
}

# 每层规则数量上限，防止 prompt 无限膨胀（节约 token 约束）
MAX_RULES_PER_LAYER = 8


def _rules_path(layer: str) -> Path:
    return LAYER_RULES_PATH[layer]


def load_rules(layer: str) -> list[str]:
    """加载某层的经验规则。文件缺失/损坏时安全降级为空列表。"""
    try:
        data = json.loads(_rules_path(layer).read_text(encoding="utf-8"))
        rules = data.get("rules", [])
        return [r.strip() for r in rules if r and r.strip()]
    except Exception:
        return []


def save_rules(layer: str, rules: list[str]) -> None:
    """保存某层规则（自动去重、裁剪到上限）。"""
    seen: list[str] = []
    for r in rules:
        r = r.strip()
        if r and r not in seen:
            seen.append(r)
    seen = seen[:MAX_RULES_PER_LAYER]
    payload = {
        "version": 1,
        "layer": layer,
        "description": (
            f"{LAYER_LABELS.get(layer, layer)}的经验库：从真实对话中总结出的优化原则，"
            "注入该层 prompt 以持续迭代（「在线 Prompt 优化反馈回路」的载体）。"
        ),
        "rules": seen,
    }
    _rules_path(layer).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def rules_suffix(layer: str) -> str:
    """把某层经验规则拼成追加到 system prompt 的片段；无规则时为空串。"""
    rules = load_rules(layer)
    if not rules:
        return ""
    lines = ["", "此外，请务必遵守以下从历史真实对话中总结出的优化原则（优先级高于上面的通用描述）："]
    lines += [f"- {r}" for r in rules]
    return "\n".join(lines)
