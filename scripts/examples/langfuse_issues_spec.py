# Langfuse 可观测接入 — 三层 issue 规格（示例，用 scripts/create_three_layer_issues.py 落库）
# 运行：uv run python scripts/create_three_layer_issues.py --spec scripts/examples/langfuse_issues_spec.py

SPEC = {
    "repo": "b1tzer/cognia",
    "requirement": {
        "title": "Cognia 接入 Langfuse 可观测：看清每一步的 prompt",
        "labels": ["kind/requirement", "status/todo"],
        "body": """## 背景
产品需要「看清每一次对话每一步的 prompt」。纯 JSON 日志堆满无关状态字段、难以查阅。
行业调研确认 Langfuse（MIT 开源、层级 trace 树、剥离 OTel 传输字段、卡片式输入/输出分 tab、可自托管/可开关）为最优解。
`.env` 已预留 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` 三变量（注释「可观测（Langfuse 开源，可选）」），本需求直接复用该约定。

## 方案概述
LangChain request-time callback 自动向下传播到所有子 runnable（含 tool 节点），主 agent 每轮 LLM + 9 个工具节点 span 零代码自动记录。
真正「差一层」的是工具内部的核心子 LLM 调用（诊断 / 探针 / 讲解 / 知识建模 / wiki 总结），它们直接 `model.invoke(...)` 未透传 `config["callbacks"]`——必须手动把 `config` 透传进这些内层调用，才能真正做到「每一步的 prompt」。

接入点：新建 `cognia/observability.py` 提供全局 `LANGFUSE_HANDLER` 单例（仅当三变量齐全才构建，否则为 None = 零侵入默认关闭）；在 `server.py` 的 AG-UI endpoint 注入 `config["callbacks"]` + 会话元数据。
透传链路：`server.py` → `tools.py` 的 `@tool`（已自带 `config: RunnableConfig`）→ `learning_engine.py` / `wiki_summarize.py` → `models.py` 的 `structured_output` / `invoke_structured` → `model.invoke(..., config)`。

## 用例清单
- [用例] 完整对话 trace 树（主 agent + 工具节点 + 工具内部子 LLM 节点全部可见）
- [用例] 默认关闭（三变量留空时零侵入、功能与未接入前一致）
- [用例] 诊断降级不丢 trace（诊断 502 降级路径仍正常记录）
- [用例] 会话按 user_id / session_id 回放与过滤
- [用例] 无回归（既有 pytest 全绿 + 新增透传不破坏 ScriptedLLM 测试桩）

## 关键决策
1. 选型 Langfuse（开源 / MIT、好看、可自托管），非 LangSmith（闭源）。
2. 开关语义：默认关闭（三变量空 → 不注入任何 callback，生产无感知）；填了才启用。
3. 不另起环境变量名，复用 `.env` 既有 `LANGFUSE_*` 三变量。
4. 不引入额外 JSON 落盘，trace 全在 Langfuse UI 呈现。
5. 透传不破坏现有安全降级链路（`propose_diagnosis` 的 `try/except` 降级不受影响）。

## Out of Scope
- Prompt 版本管理（Langfuse Prompt Management）接入——本期只做追踪，不做 prompt CMS。
- 在线 prompt 优化反馈回路（已有独立需求），本期不耦合。
- 采样策略、PII 脱敏——本期先用全量记录，后续按需加。

## DoD
- 代码通过 lint + 全量 pytest
- 改动仅限上述 7 处文件 + `observability.py` 新建，不夹带重构
- `.env` / 部署文档说明开关方式
- 至少 1 次真实对话在 Langfuse UI 肉眼核对层级树与 prompt 可读性""",
    },
    "use_cases": [
        {
            "title": "完整对话 trace 树：主 agent + 工具节点 + 工具内部子 LLM 节点全部可见",
            "labels": ["kind/use-case", "status/todo"],
            "body": """## 用户故事
作为 Cognia 的开发者 / PM，当我跑一段包含「构建目标 → 诊断 → 讲解」的完整对话时，我希望在 Langfuse 看到一棵层级 trace 树，其中主 agent 每轮 LLM 节点、每个工具节点、以及**工具内部**的诊断 / 探针 / 讲解 / 知识建模 LLM 节点都清晰可见，点开任意 generation 能看到完整 input prompt 与 output、token、cost、latency，以便排查 prompt 效果与行为，而不是在纯 JSON 里翻状态字段。

## 验收标准（AC）
### 正常
- AC1.1 `.env` 三变量填好后，跑一段含「诊断 → 讲解 → 构建目标」的完整对话，Langfuse 出现 1 条 trace。
- AC1.2 树中能看到主 agent 每轮 ReAct teacher LLM 节点（input prompt 完整）。
- AC1.3 树中能看到 9 个工具节点 span。
- AC1.4 树中能看到工具**内部**的子 LLM 节点：诊断（run_diagnosis）、探针（generate_probe）、讲解（explain）、知识建模（build_knowledge_model）——这些节点嵌套在对应工具节点下，点开可见完整 system + user + output。
- AC1.5 任意 generation 节点能并排看到 model、input tokens、output tokens、cost、latency。
- AC1.6 无纯 JSON 噪音：Langfuse 前台只展示关心的输入 / 输出 / 元数据，底层 OTel 传输字段已被剥离。

### 边界
- AC1.7 对话不含任何工具调用（纯闲聊）时，trace 仅含主 agent 的 LLM 节点，不报错。

### 异常
- （见「诊断降级不丢 trace」用例）""",
        },
        {
            "title": "默认关闭：三变量留空时零侵入、功能与未接入前一致",
            "labels": ["kind/use-case", "status/todo"],
            "body": """## 用户故事
作为部署 Cognia 的运维，当我在 `.env` 把三变量留空（默认状态）时，我希望服务照常启动、对话功能与接入 Langfuse 之前完全一致，没有任何 Langfuse 网络请求、没有延迟、没有报错，以便生产环境默认不受可观测组件影响。

## 验收标准（AC）
### 正常
- AC2.1 三变量留空时，`cognia/observability.py` 的 `LANGFUSE_HANDLER` 为 None，endpoint 不注入任何 callbacks。

### 边界
- AC2.2 仅填其中 1~2 个变量（其余空）时，仍视为关闭（不连接），不抛「缺 key」异常。

### 异常
- AC2.3 三变量留空时，跑一段完整对话，行为、输出、延迟与接入前 pytest / 手测基线完全一致（无回归）。
- AC2.4 三变量留空时，确认运行期无任何发往 `LANGFUSE_HOST` 的 HTTP 请求。""",
        },
        {
            "title": "诊断降级不丢 trace：诊断 502 降级路径仍正常记录",
            "labels": ["kind/use-case", "status/todo"],
            "body": """## 用户故事
作为开发者，当诊断模型上游持续 502（MaaS 不可用）导致 `propose_diagnosis` 走安全降级（catch 后返回 recorded=False 信号）时，我希望 Langfuse 中该工具节点仍正常记录（标记降级结果），trace 不被丢弃、SSE 流不被中断，以便事后能复盘「哪次对话因诊断失败走了降级」。

## 验收标准（AC）
### 正常
- AC3.1 诊断正常时，propose_diagnosis 节点记录 diagnosed_state / confidence / evidence / recorded=True 等字段。

### 边界
- （无特别边界）

### 异常
- AC3.2 诊断模型 502 时，propose_diagnosis 走 try/except 降级，trace 中该工具节点仍正常记录，output 含 `recorded: false` 与 error 说明，不丢 trace。
- AC3.3 降级路径下 SSE 流不被中断（前端不出现 SocketError / INCOMPLETE_STREAM），可观测接入不引入新异常。
- AC3.4 透传 config 不破坏现有降级逻辑（config=None 或 handler=None 时 try/except 行为不变）。""",
        },
        {
            "title": "会话按 user_id / session_id 回放与过滤",
            "labels": ["kind/use-case", "status/todo"],
            "body": """## 用户故事
作为 PM，我希望在 Langfuse 中能按 `user_id`（匿名标识）和 `session_id`（thread_id）过滤、回放单次学习会话的全部步骤，以便针对某个具体用户某次对话做复盘与 prompt 调优，而不是在全局 trace 流里手动翻找。

## 验收标准（AC）
### 正常
- AC4.1 endpoint 注入 `metadata.langfuse_user_id = user_id`、`metadata.langfuse_session_id = thread_id`，Langfuse 可按二者过滤。
- AC4.2 选某 session_id 后，能回放该会话完整 trace 树（含所有嵌套子 LLM 节点），顺序与真实对话一致。

### 边界
- AC4.3 user_id 缺失降级为 "local-user" 时，该值仍正确写入 metadata（与 server.py 现有降级一致）。

### 异常
- （无特别异常）""",
        },
        {
            "title": "无回归：既有 pytest 全绿 + 新增透传不破坏 ScriptedLLM 测试桩",
            "labels": ["kind/use-case", "status/todo"],
            "body": """## 用户故事
作为维护者，当我给核心编排 / 纯函数新增 `config` 透传形参时，我希望既有单元测试（尤其内存对象、不连 LLM / DB 的 ScriptedLLM 测试桩，以及对 `invoke_structured` / `structured_output` / `build_knowledge_model` 的直接调用）全部保持通过，新增形参默认 None 时行为完全不变，以便可观测接入零回归风险。

## 验收标准（AC）
### 正常
- AC5.1 `pytest` 全量通过（含现有 learning_engine / models / tools 测试）。
- AC5.2 `config=None` 时，`structured_output` / `invoke_structured` / `run_diagnosis` / `build_knowledge_model` / `wiki_summarize._summarize` 行为与接入前逐字节一致（不传 config 即原路径）。

### 边界
- AC5.3 既有对 `build_knowledge_model` 等函数的测试若未传 config，调用不报错（形参有默认值 None）。

### 异常
- AC5.4 lint 通过（无未使用变量、类型注解一致）。""",
        },
    ],
    "tasks": [
        {
            "title": "T1 observability 开关层：新建 cognia/observability.py + pyproject.toml 加 langfuse 依赖",
            "labels": ["kind/task", "status/todo", "area/backend", "assignee/pm"],
            "body": """## 契约
**做什么**：新建 `cognia/observability.py` 提供全局 `LANGFUSE_HANDLER` 单例；在 `pyproject.toml` 的 dependencies 追加 `langfuse` 依赖。
**涉及文件 / 接口**：
- 新建 `cognia/observability.py`：读取 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`，三者齐全时 `LANGFUSE_HANDLER = CallbackHandler()`，否则 `None`（from langfuse.langchain import CallbackHandler）。
- `pyproject.toml`：dependencies 追加 `"langfuse>=3.0.0,<4.0.0"`。

**任务级验收标准**
- T1.1 `pyproject.toml` 含 langfuse 依赖，`uv sync` / `pip install` 可装。
- T1.2 三变量留空时 `LANGFUSE_HANDLER is None`；三者齐全时为 CallbackHandler 实例（不抛缺 key 异常）。
- T1.3 仅填 1~2 个变量时仍为 None（视为关闭）。
- T1.4 导入 `cognia.observability` 无副作用（不连接网络）。""",
        },
        {
            "title": "T2 server 注入层：server.py endpoint 注入 config（callbacks + metadata）",
            "labels": ["kind/task", "status/todo", "area/backend", "assignee/pm"],
            "body": """## 契约
**做什么**：在 `cognia/server.py` 的 `cognia_agent_endpoint` 中，给 `LangGraphAGUIAgent` 注入 `config["callbacks"]` + 会话元数据。
**涉及文件 / 接口**：
- `cognia/server.py`：import `LANGFUSE_HANDLER` from `cognia.observability`；构造 `request_agent` 时合并 `{"configurable": {"user_id": user_id}}` 与（若 handler 非 None）`{"callbacks": [LANGFUSE_HANDLER], "metadata": {"langfuse_user_id": user_id, "langfuse_session_id": input_data.thread_id or ""}}`。

**任务级验收标准**
- T2.1 handler 非 None 时，config 含 callbacks + metadata；为 None 时不注入（功能不变）。
- T2.2 确认 CopilotKit 的 `LangGraphAGUIAgent` 将 `config["callbacks"]` 透传给 graph（与现有 `configurable.user_id` 透传路径一致）；若被丢弃，fallback：在 `build_agent()` 内对 graph 调用 `.with_config({"callbacks": [h]})` 原地绑定。
- T2.3 服务启动 + 对话功能与基线一致（pytest / 手测无回归）。""",
        },
        {
            "title": "T3 models 透传层：models.py structured_output / invoke_structured 加 config=None 透传",
            "labels": ["kind/task", "status/todo", "area/backend", "assignee/pm"],
            "body": """## 契约
**做什么**：`cognia/models.py` 的 `structured_output` 与 `invoke_structured` 增加 `config=None` 形参并透传到 `model.invoke(...)`。
**涉及文件 / 接口**：
- `structured_output(model, schema, messages, config=None)`：透传到 `model.with_structured_output(schema, tool_choice="auto").invoke(messages, config)` 及降级分支 `invoke_structured(model, schema, messages, config)`。
- `invoke_structured(model, schema, messages, config=None)`：`model.invoke([*messages, ("human", instruction)], config)`。

**任务级验收标准**
- T3.1 `config=None` 时行为与接入前逐字节一致（不传 config = 原路径）。
- T3.2 `config` 非 None 时透传到底层 `model.invoke` / `.invoke(messages, config)`。
- T3.3 既有对这两个函数的调用（含 ScriptedLLM 桩）不传 config 仍正常。""",
        },
        {
            "title": "T4 learning_engine 透传层：run_diagnosis / build_knowledge_model 加 config=None 透传",
            "labels": ["kind/task", "status/todo", "area/backend", "assignee/pm"],
            "body": """## 契约
**做什么**：`cognia/learning_engine.py` 的 `run_diagnosis` 与 `build_knowledge_model` 增加 `config=None` 形参并透传。
**涉及文件 / 接口**：
- `run_diagnosis(diagnoser, point, question, user_answer, config=None)` → `models.structured_output(diagnoser, Diagnosis, [...], config=config)`。
- `build_knowledge_model(planner, goal, config=None)` → 两处 `planner.invoke(messages, config)`（首调 + 解析失败重试调）。

**任务级验收标准**
- T4.1 `config=None` 时行为与接入前一致。
- T4.2 `config` 非 None 时透传到 `planner.invoke` / `structured_output`。
- T4.3 重试分支（JSON 解析失败回喂）也透传 config。""",
        },
        {
            "title": "T5 tools 透传层：propose_diagnosis / generate_probe / explain / build_learning_goal / summarize_session_to_wiki 透传 config",
            "labels": ["kind/task", "status/todo", "area/backend", "assignee/pm"],
            "body": """## 契约
**做什么**：`cognia/tools.py` 中现有 `@tool` 函数已自带 `config: RunnableConfig`（LangChain 注入），把该 config 透传给工具内部的 LLM 调用。
**涉及文件 / 接口**：
- `propose_diagnosis`：`run_diagnosis(_get_diagnoser(), point, question, user_answer, config)`。
- `generate_probe`：`_get_teacher().invoke([("system", PROBE_GENERATOR_SYSTEM_PROMPT), ("human", ...)], config=config)`。
- `explain`：`_get_teacher().invoke([...], config=config)`。
- `build_learning_goal`：`build_knowledge_model(_get_planner(), goal, config)`。
- `summarize_session_to_wiki`：`summarize_thread_to_wiki(..., config=config)`。

**任务级验收标准**
- T5.1 上述 5 个工具的内部 LLM 调用均透传 config。
- T5.2 保留 `propose_diagnosis` 现有 try/except 降级（config=None 或 handler=None 时降级行为不变）。
- T5.3 不触碰 web_search / query_proficiency（它们不调 LLM）。""",
        },
        {
            "title": "T6 wiki 透传层：wiki_summarize._summarize 加 config=None 透传",
            "labels": ["kind/task", "status/todo", "area/backend", "assignee/pm"],
            "body": """## 契约
**做什么**：`cognia/wiki_summarize.py` 的 `_summarize` 增加 `config=None` 形参并透传。
**涉及文件 / 接口**：
- `_summarize(model, dialogue, title_hint=None, config=None)` → `models.invoke_structured(model, WikiSummary, [...], config)`。
- `summarize_thread_to_wiki(...)` 调用 `_summarize` 处透传 `config`（从 tools.py 的 `summarize_session_to_wiki` 传入）。

**任务级验收标准**
- T6.1 `config=None` 时行为与接入前一致。
- T6.2 `config` 非 None 时透传到 `invoke_structured`。""",
        },
        {
            "title": "T7 部署文档 / 脚本：Langfuse Cloud 或自托管 docker-compose 接入说明",
            "labels": ["kind/task", "status/todo", "area/backend", "assignee/pm"],
            "body": """## 契约
**做什么**：编写 Langfuse 接入部署说明（`.env` / README 或 `docs/` 片段），覆盖 Cloud 与自托管两条路。
**涉及文件 / 接口**：
- 在 `.env` 的 LANGFUSE_* 三变量注释补充使用方法；或新增 `docs/observability-langfuse.md`。
- 覆盖：①Langfuse Cloud 免费 Hobby（注册拿 key 填三变量即可）；②自托管 `git clone https://github.com/langfuse/langfuse && docker compose up -d`，`LANGFUSE_HOST` 指向本机；③开关语义（留空 = 关闭）。

**任务级验收标准**
- T7.1 文档说明三变量如何获取、如何填。
- T7.2 文档说明默认关闭语义与自托管起停命令。
- T7.3 不引入脚本之外的运行时依赖变更（docker-compose 仅为可选自托管指引）。""",
        },
    ],
}
