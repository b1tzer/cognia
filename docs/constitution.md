# Cognia Constitution（宪法）

> 本文档是 Cognia 项目**不可谈判的规则**。任何功能开发、AI 生成代码、重构，都必须遵守本文。若与本文冲突，以本文为准；如需变更，必须先修改本文并评审通过。

## 0. 项目定位

- 名称：Cognia
- 一句话：AI 主动学习教练，主动发现用户认知盲区并动态引导掌握知识
- 初始用户：程序员学习新框架 / 编程语言
- 北极星指标：AI 主动诊断的认知缺陷中，≥70% 被用户承认、≥50% 与专家标注命中

## 1. 架构风格

- 分层：表现层（UI）→ 编排层（LangGraph）→ 记忆层（Checkpointer / Store）→ 模型层（LLM）
- 状态优先：所有多步逻辑都必须建模为 LangGraph 的 Stateful Graph
- 单向数据流：节点之间通过 State 传递，节点不直接互相调用
- 副作用隔离：有副作用的操作（写库、调用外部 API）只允许放在 tool 节点

## 2. 技术栈决策

| 层面 | 决策 | 理由 |
|------|------|------|
| 语言 / 运行时 | Python 3.12+ | LangGraph 主生态语言，agent 开发事实标准 |
| Agent 编排 | LangGraph 1.x | 已锁定（第一步调研结论） |
| LLM | **DeepSeek**，走 LangChain 集成保持模型无关 | 国内访问顺畅；集成层解耦，可随时替换 OpenAI / Anthropic / 本地模型 |
| 数据库 | **Supabase 托管**（底层 Postgres + pgvector） | 用空间换时间，优先保证「快」；一个库同时承担 Checkpointer + Store + 向量检索 |
| 向量检索 | pgvector（Supabase 内嵌） | 避免引入独立向量库的运维成本 |
| Web 框架 | FastAPI | 异步、快、现代、生态好 |
| 前端（MVP） | **Chainlit 或 Streamlit** | Python 极速 UI 验证，确保能体验真实流式对话体感，不走纯脚本 |
| 可观测 | Langfuse（开源）优先，LangSmith 备选 | 符合开源偏好；LangSmith 付费层后续再上 |
| 评估 | 自建金标集 + 脚本，可借用 LangChain evals | 北极星要求专家标注集，自建最贴合 |

## 3. Agent 编排原则（不可谈判）

- 拓扑默认 supervisor 模式（拆解 → 派 worker → 汇总），禁用 swarm / peer-to-peer
- 能单干就不拆 agent；仅当满足以下至少一条才拆分：
  ① 任务天然可并行 ② 需要明显不同专长 ③ 单 agent 上下文装不下
- agent 总数上限 4 个（超过则协调税吃掉分工收益）
- 上下文按需传递，不广播完整历史；结构化交接只传当前步骤需要的字段
- 大产物传引用不传拷贝；每个 sub-agent 拿到自包含任务描述 + 期望输出格式
- 共享记忆分层：私有命名空间 + 共享层按需检索，不整板注入

## 4. 安全红线（违反即阻塞上线）

- 所有用户输入在 API 边界校验，禁止信任客户端数据
- 密钥 / 凭证一律环境变量注入，严禁硬编码进仓库
- 日志 / 追踪中禁止输出完整 prompt 中的用户敏感信息（脱敏）
- 记忆跨租户访问必须经过认证层鉴权，namespace 只是作用域不是授权

## 5. 记忆与个性化规范

- 三层记忆：短期（thread 级 Checkpointer）/ 摘要（可选）/ 长期（Store）
- 多租户隔离：namespace 结构性隔离，禁止「单 namespace + metadata 事后过滤」
- 身份安全：user_id 必须来自认证层，严禁从客户端请求参数读取
- 用户画像：结构化 Profile 增量更新，注入 system prompt
- 身份注入：user_id / 模型服务句柄走 runtime context，不塞进 State
- **认知 Profile 进化**：用户 Profile 须区分「基础偏好」（沟通风格、语言、时区等稳定属性）与「知识熟练度 JSON」（对具体知识点的掌握状态，随学习动态变化）；两类数据都必须采用**增量 Delta 更新**，严禁全量重写，防止记忆断层

## 6. 测试与质量

- 每个新增节点 / 条件边必须带单元测试
- 核心诊断逻辑必须有金标数据集回归（北极星指标）
- 变更 prompt / tool / graph 前，先跑既有 eval 不回归
- 禁止 `# TODO 以后再测` 进入主分支
- **金标集留存隔离**：测试集须区分「开发可见集」与「盲测集」；禁止将盲测 Case 作为 Few-shots 写入 Prompt，否则会导致过拟合

## 7. 代码规范

- 遵循 PEP 8；类型注解用 TypedDict + Pydantic
- 注释用中文（见项目语言偏好）
- State 用 TypedDict 定义；累加字段必须配 reducer
- 节点保持纯函数，尽量小（单节点 ≤ 150 行）
