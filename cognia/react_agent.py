"""Cognia 教学 Agent system prompt 兼容入口。

原 REACT_TEACHER_SYSTEM_PROMPT 文本已抽离到 cognia.prompts.teacher
（prompts-as-code），本文件仅保留旧名 re-export，供 server.py 等
既有 import 路径零改动过渡。
"""

from cognia.prompts.teacher import TEACHER_SYSTEM_PROMPT

# 兼容旧名：server.py 现役 import 路径
REACT_TEACHER_SYSTEM_PROMPT = TEACHER_SYSTEM_PROMPT

__all__ = ["REACT_TEACHER_SYSTEM_PROMPT"]
