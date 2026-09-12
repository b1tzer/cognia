# Cognia

一个 **AI Learning Agent**：它不只是把知识告诉用户，而是主动理解用户「到底懂了什么、没懂什么、哪里理解错了」，并通过动态对话帮助用户建立完整、正确、可迁移的知识体系。

> 产品哲学：**慢就是快，快就是慢** —— 宁慢也要让认知真正建立，拒绝技巧灌输。

## 核心思路

Cognia 持续构建两套模型，并根据二者差距主动驱动学习：

- **知识模型（Knowledge Model）**：围绕学习目标生成的 DAG 概念图，表达概念间的因果与依赖关系。
- **用户认知模型（Cognitive Model）**：对每个概念的掌握状态建模（未探索 / 半懂 / 误解 / 掌握）。

学习闭环如下：

```
学习目标
  → 自动构建知识模型
  → 推断「真正理解需要什么」
  → 让用户主动表达理解
  → 认知诊断（BKT 贝叶斯知识追踪）
  → 发现：理解 / 半理解 / 误解 / 信息不足
  → 教学决策：追问 / 解释 / 纠错 / 回溯 / 继续
  → 再次验证 → 持续更新认知模型
  → 形成完整、正确、可迁移的理解
```

核心机制详见 [Purpose.md](./Purpose.md)。

> 协作与开发规范：**通用需求交付元流程**（调研 → 设计 → 开发 → 验收）见 [docs/general-sop.md](./docs/general-sop.md)；本项目的具体操作 SOP（AAR 复盘 / GitHub issue 协作 / 前后端分工 / 测试验收等）见 [docs/SOP.md](./docs/SOP.md)。

## 技术栈

| 层 | 技术 |
| --- | --- |
| 后端 | Python + FastAPI + Uvicorn |
| 前端 | React + TypeScript + Vite |
| 存储 | SQLite（`backend/cognia.db`） |
| AI 接入 | OpenAI 兼容协议（openai SDK + `chat.completions`），不绑定任何具体模型/供应商 |
| 认知诊断 | BKT（贝叶斯知识追踪） |

## 目录结构

```
cognia/
├── backend/            # FastAPI 后端
│   ├── main.py         # 应用入口与学习闭环编排
│   ├── domain_model.py # 知识模型构建
│   ├── cognitive.py    # 认知诊断（BKT 贝叶斯更新）
│   ├── tutor.py        # 教学决策与苏格拉底式回复
│   ├── llm.py          # OpenAI 兼容 LLM 封装
│   ├── db.py           # SQLite 会话/消息存储
│   ├── config.py       # 全局配置
│   ├── schemas.py      # Pydantic 数据模型
│   ├── eval/           # 诊断规则与评测样本
│   └── .env.example    # 配置模板
├── frontend/           # React 前端
│   └── src/
│       ├── App.tsx
│       ├── api.ts
│       └── components/ # ChatPanel / KnowledgeGraph / CognitivePanel / ...
├── start.sh            # 一键启动脚本
└── Purpose.md          # 产品目标说明
```

## 快速开始

### 一键启动（推荐）

```bash
./start.sh
```

脚本会自动完成：后端虚拟环境初始化与依赖安装 → 前端依赖安装与构建 → 启动服务。启动后访问 <http://localhost:8000>。

### 手动启动（开发模式）

**后端：**

```bash
cd backend
python3.12 -m venv .venv          # 推荐 3.11 ~ 3.13
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

**前端（开发热更新，端口 5173，`/api` 已代理到后端）：**

```bash
cd frontend
pnpm install
pnpm dev
```

## 配置

后端通过环境变量或 `backend/.env` 覆盖默认值，模板见 [backend/.env.example](./backend/.env.example)：

```bash
cp backend/.env.example backend/.env
```

关键配置项：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `OPENAI_BASE_URL` | OpenAI 兼容接口地址 | `http://127.0.0.1:8090/v1` |
| `OPENAI_API_KEY` | 鉴权 Key | `tdai-key` |
| `OPENAI_MODEL` | 主模型 | `deepseek-v4-flash` |
| `OPENAI_MODEL_FALLBACK` | 主模型失败时的备选模型 | `glm-5-3-flash-internal` |
| `COGNIA_AI_ENABLED` | `1` 启用 AI 引擎，`0` 强制离线诊断模式 | `1` |
| `COGNIA_HOST` | 监听地址 | `0.0.0.0` |
| `COGNIA_PORT` | 服务端口 | `8000` |

> **说明**：Cognia 只依赖 OpenAI 兼容协议这一事实标准。交付/生产环境只需把标准 OpenAI 的 `base_url` 与 `api_key` 填入配置即可，代码零改动。默认的 `127.0.0.1:8090` adapter 仅用于本地开发，属临时资源，并非交付标准。

## 主要 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 健康检查，返回 AI 引擎状态与模型名 |
| `POST` | `/api/sessions` | 创建学习会话（传入 `goal`） |
| `GET` | `/api/sessions` | 会话列表 |
| `GET` | `/api/sessions/{sid}` | 会话详情 |
| `POST` | `/api/sessions/{sid}/chat` | 发送消息，返回诊断结果、教学动作与回复 |

前端构建产物由后端 `main.py` 静态托管（`/assets` 与 SPA 回退），因此一键启动后前后端统一由 8000 端口对外服务。

## UI 视觉方向

「**Cognitive Atlas 认知图谱** —— 一张随理解点亮的安静星图」：低饱和中性底 + 墨色 + 单一强调色，知识模型呈 DAG 点线星图，认知状态映射为节点明暗/填充度（未探索=虚点、半懂=缺口、误解=偏色、掌握=实心点亮）。
