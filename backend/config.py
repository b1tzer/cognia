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
# 掌握阈值：超过该概率视为"已掌握"（对话式辅导折中，非练习系统的 0.95）
MASTERY_THRESHOLD = 0.85
# 掌握判定所需的最小独立答对次数（证据充分性，防止一问即判）
MIN_SUCCESS_EVIDENCE = 2

# --------------------------------------------------------------------------
# 教学决策引擎（分层流程控制）参数
# 落地「确定性骨架 + LLM 语义决策」混合架构：
# 骨架层写死安全不变量，LLM 只在候选集内做语义选择。
# --------------------------------------------------------------------------
# 单概念最大交互步数（防死循环安全网：超过则强制推进到下一概念）
# 对齐提示词「一个知识节点最多 1~2 次验证」：默认向前，纠缠是例外。
MAX_STEPS_PER_CONCEPT = 2
# 停滞检测窗口：连续 N 轮诊断状态无变化则强制切换策略
STAGNATION_WINDOW = 3
# 最近发展区（ZPD）：焦点概念的目标预测成功率窗口
ZPD_MIN = 0.60
ZPD_MAX = 0.85
# 回溯触发：同一概念连续失败次数达到该值则回溯到前置概念
BACKTRACK_CONSECUTIVE_FAILURES = 3
# 回溯触发：前置概念掌握度衰退到该值以下则回溯巩固
BACKTRACK_MASTERY_FLOOR = 0.50
# 诊断置信度下限：低于该值宁可降级为 partial 并追问，不强判
DIAG_CONFIDENCE_FLOOR = 0.70
# 教学决策候选集最大数量（控制 LLM 决策输入 token）
DECISION_MAX_CANDIDATES = 3

# --------------------------------------------------------------------------
# 显式 token 预算（Phase 2：把散落的 max_tokens 收敛到一处）
#
# 落地行业最佳实践（Anthropic context engineering / TokenBudget 类模式）：
# - prompt_tokens：该层输入（system+user）的 token 上限，超限触发告警（不静默失败）
# - completion_tokens：该层输出（即 chat 的 max_tokens）上限
# - 数值对齐当前各层实际规模，保证「行为等价」（prompt 上限宽松，仅用于告警而非裁剪）
#
# 各层：domain_model（知识模型构建）/ cognitive（认知诊断）/ focus_select（焦点选择）
#       decision_action（教学动作决策）/ tutor（回复生成）
# ---------------------------------------------------------------------------
CONTEXT_BUDGET = {
    "domain_model":    {"prompt_tokens": 2000, "completion_tokens": 4000},
    "cognitive":       {"prompt_tokens": 4000, "completion_tokens": 1500},
    "focus_select":    {"prompt_tokens": 2000, "completion_tokens": 1000},
    "decision_action": {"prompt_tokens": 2500, "completion_tokens": 1000},
    "tutor":           {"prompt_tokens": 4000, "completion_tokens": 1500},
}
# 上下文利用率告警阈值：超过该比例触发告警（行业经验 60-85% 为最优区间）
BUDGET_WARN_RATIO = 0.80

# --------------------------------------------------------------------------
# 认知状态四分类的置信度区间（用于诊断结果可视化）
# --------------------------------------------------------------------------
# 掌握概率 -> 状态映射
STATE_BANDS = {
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

# --------------------------------------------------------------------------
# 在线 Prompt 优化反馈回路（慢循环）
# --------------------------------------------------------------------------
# 总开关：是否启用后台在线优化
OPTIMIZER_ENABLED = os.getenv("COGNIA_OPTIMIZER_ENABLED", "1") != "0"
# 检查节拍（秒）：定时器作为「检查水位」的节拍，而非直接执行优化的节拍
OPTIMIZER_INTERVAL_SECONDS = int(os.getenv("COGNIA_OPTIMIZER_INTERVAL", "600"))
# 数据增量触发阈值：自上次优化以来「新增对话轮次」达到该值才真正执行优化，
# 未达到则本轮什么都不做（不调用任何 LLM，零 token 浪费）
OPTIMIZER_MIN_NEW_TURNS = int(os.getenv("COGNIA_OPTIMIZER_MIN_NEW_TURNS", "10"))
# 每层每轮最多采样的样本数（节约 token）
OPTIMIZER_MAX_SAMPLES_PER_LAYER = int(os.getenv("COGNIA_OPTIMIZER_MAX_SAMPLES", "20"))
# 单层样本下限：某层可提取的新样本数低于此值则跳过该层（二次保护）
OPTIMIZER_MIN_SAMPLES = int(os.getenv("COGNIA_OPTIMIZER_MIN_SAMPLES", "5"))
# 水位状态文件：记录各会话已消费到的消息位置，避免重复消费旧数据
OPTIMIZER_STATE_PATH = os.getenv("COGNIA_OPTIMIZER_STATE", str(BASE_DIR / "optimizer_state.json"))
