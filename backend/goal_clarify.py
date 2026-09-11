"""学习目标澄清模块。

在构建知识模型之前，先判断用户提出的学习目标是否清晰、无歧义：
- 目标清晰 → 直接进入知识模型构建（need_clarify=False）
- 目标存在歧义/同名概念（如「IO」可能是网络IO/磁盘IO/Java NIO/内核IO模型）→
  输出候选理解与澄清提问，让用户确认后再开始（need_clarify=True）

同时提供「中途修改目标」的意图识别（确定性规则，零 token），与显式入口
共同支撑「理解错误时可随时修改目标」的产品要求。

架构遵循「规则优先、LLM 增强」：LLM 判歧义，规则兜底（目标过短/AI 不可用）。
"""
from __future__ import annotations

from typing import Optional

import config
from llm import chat_json


# 中途修改目标的意图信号词（确定性规则，零 token）
_GOAL_CHANGE_HINTS = (
    "换个目标", "改目标", "修改目标", "换目标", "换一个目标", "重新学", "重新开始",
    "我想学的是", "其实我想学", "其实想学", "我想学", "想学的是",
    "我指的是", "指的是", "理解错了", "理解偏了", "不是这个",
    "换一个方向", "换方向", "目标改成", "目标改为",
)


_CLARIFY_SYSTEM = """你是 Cognia 的学习目标澄清助手。用户提出了一个学习目标，你的任务是判断它是否足够清晰、无歧义，足以开始构建学习路径。

规则：
1. 目标清晰无歧义（如「Git 版本控制」「Python 异常处理」）→ 输出 {"need_clarify": false, "goal": "规范化后的目标描述"}
2. 目标存在歧义或同名概念（如「IO」可能是网络IO/磁盘IO/Java NIO/操作系统IO模型；「并发」可能是多线程/协程/分布式并发；「机器学习」范围过宽）→ 输出 {"need_clarify": true, "candidates": ["候选理解1","候选理解2"], "question": "一句话澄清提问"}
3. 目标含糊到无法判断方向（如「学技术」）→ 输出 {"need_clarify": true, "candidates": [], "question": "引导用户说得更具体的提问"}

额外要求：
- 候选理解 candidates 最多 3~4 个，彼此区分明显，覆盖最常见的几种解读。
- candidates 里每个候选都要是完整、可直接作为学习目标使用的短语（保留限定词）。
- 只输出 JSON，不要任何解释、思考过程或多余文字。
"""


def clarify_goal(goal: str, trace: Optional[list] = None) -> dict:
    """判断学习目标是否需要澄清。

    返回 dict：
    {
        "need_clarify": bool,     # 是否需要澄清
        "goal": str,              # 规范化后的目标（不澄清时使用）
        "candidates": list[str],  # 候选理解（澄清时提供给用户点选）
        "question": str,          # 澄清提问（澄清时展示）
    }
    """
    g = (goal or "").strip()

    # 规则兜底 1：目标过短，无法判断方向
    if len(g) < 4:
        return {
            "need_clarify": True,
            "goal": g,
            "candidates": [],
            "question": "这个目标有点简短，能再具体一点吗？比如你想深入理解的具体主题或技术方向是什么？",
        }

    # 规则兜底 2：AI 不可用 → 不澄清，直接构建
    if not config.AI_ENABLED:
        return {"need_clarify": False, "goal": g, "candidates": [], "question": ""}

    data = chat_json(
        _CLARIFY_SYSTEM,
        f"学习目标：{g}",
        temperature=0.2,
        max_tokens=800,
        trace=trace,
        trace_label="需求澄清",
    )
    if not data:
        # LLM 失败 → 不澄清，直接构建（避免阻塞主流程）
        return {"need_clarify": False, "goal": g, "candidates": [], "question": ""}

    need = bool(data.get("need_clarify", False))
    candidates = [str(c).strip() for c in (data.get("candidates") or []) if str(c).strip()]
    question = str(data.get("question") or "").strip()
    normalized = str(data.get("goal") or g).strip()

    # 即便 need_clarify=True，也要有提问或候选，否则降级为不澄清
    if need and not question and not candidates:
        return {"need_clarify": False, "goal": g, "candidates": [], "question": ""}

    return {
        "need_clarify": need,
        "goal": normalized or g,
        "candidates": candidates,
        "question": question,
    }


def detect_goal_change_intent(user_text: str) -> bool:
    """检测用户是否表达「修改学习目标」的意图。

    仅命中信号词时触发；返回 True 表示本轮应走「改目标」分支而非正常诊断。
    """
    t = (user_text or "").strip()
    if not t:
        return False
    return any(h in t for h in _GOAL_CHANGE_HINTS)
