"""Cognia 全局配置。

所有可调参数集中于此，支持通过环境变量或 backend/.env 覆盖。
AI 相关配置遵循 OpenAI 兼容协议，可对接 OpenAI / DeepSeek / Ollama 等。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --------------------------------------------------------------------------
# AI 模型（OpenAI 兼容协议）
#
# 默认对齐本机 TencentDB Agent Memory 的接入方式：走宿主机 8090 端口的
# adapter.py（OpenAI 兼容代理），由它转发到工蜂 Copilot Gateway。
#   - OPENAI_BASE_URL 指向本机 adapter
#   - OPENAI_API_KEY  只要求非空（adapter 不校验 Bearer，真实鉴权在代理层注入）
#   - OPENAI_MODEL    默认用 fast 模型（adapter 白名单直通，不会被轮换覆盖）
#   - OPENAI_MODEL_FALLBACK 主模型失败时回退的备选 fast 模型
# 若需临时关闭 AI 引擎（强制离线诊断模式），设 COGNIA_AI_ENABLED=0。
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "tdai-key")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8090/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
OPENAI_MODEL_FALLBACK = os.getenv("OPENAI_MODEL_FALLBACK", "glm-5-3-flash-internal")
AI_ENABLED = os.getenv("COGNIA_AI_ENABLED", "1") != "0"

# --------------------------------------------------------------------------
# 认知诊断参数（BKT 贝叶斯知识追踪）
# --------------------------------------------------------------------------
# 每个知识点初始掌握概率
P_L0 = 0.35
# 学习转移概率（未掌握 -> 掌握）
P_LEARN = 0.20
# 猜对概率（未掌握却答对）
P_GUESS = 0.20
# 失误概率（已掌握却答错）
P_SLIP = 0.10
# 掌握阈值：超过该概率视为"已掌握"
MASTERY_THRESHOLD = 0.80

# --------------------------------------------------------------------------
# 认知状态四分类的置信度区间（用于诊断结果可视化）
# --------------------------------------------------------------------------
# 掌握概率 -> 状态映射
STATE_BANDS = {
    "mastered": 0.80,      # 理解
    "partial": 0.55,       # 半理解
    "misconceived": 0.55,  # 错误（有明确错误证据时优先判定）
    "insufficient": 0.55,  # 信息不足（无证据或证据太少）
}

# --------------------------------------------------------------------------
# 存储
# --------------------------------------------------------------------------
DB_PATH = os.getenv("COGNIA_DB", str(BASE_DIR / "cognia.db"))

# --------------------------------------------------------------------------
# 服务
# --------------------------------------------------------------------------
HOST = os.getenv("COGNIA_HOST", "0.0.0.0")
PORT = int(os.getenv("COGNIA_PORT", "8000"))
