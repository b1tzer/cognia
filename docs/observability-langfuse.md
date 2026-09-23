# Cognia 可观测性接入（Langfuse）

> 目标：看清每一次对话每一步的 prompt —— 层级 trace 树，而非纯 JSON 噪音。

## 选型

Langfuse（MIT 开源、可自托管、UI 好看、自动剥离底层 OTel 传输字段）。
主 agent 走 LangChain，request-time callback 自动向下传播到所有子 runnable（含 tool 节点），
因此主 agent 每轮 LLM + 工具节点 span **零代码自动记录**；工具内部的核心子 LLM 调用
（诊断 / 探针 / 讲解 / 知识建模 / wiki 总结）已手动透传 `config`，也能挂到 trace 树上。

## 开关语义（零侵入）

默认**关闭**：仅当 `.env` 的 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`
**三变量齐全**时，`cognia/observability.py` 才构建 `CallbackHandler` 并注入；
否则 `LANGFUSE_HANDLER is None`，endpoint 不注入任何 callback，生产无感知、无网络请求。

## 接入步骤

### 方式一：Langfuse Cloud（最快）

1. 注册 https://cloud.langfuse.com （免费 Hobby 套餐）。
2. Project Settings → API Keys，拿到 Public / Secret Key。
3. 在 `.env` 填：
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```
4. 重启服务（`uv run python -m cognia.server`）。

### 方式二：自托管（数据不出内网）

```bash
git clone https://github.com/langfuse/langfuse
cd langfuse && docker compose up -d
```

> **本地部署实战踩坑（本机已部署，UI 在 http://localhost:3001）**
> 1. **端口冲突**：官方 compose 默认 `web:3000 / postgres:5432 / clickhouse:8123`，
>    本机已被 Cognia 前端(3000)、`cognia-pg`(5432)、其他服务(8123) 占用，需在
>    `docker-compose.yml` 把宿主映射改为 `web 3001 / postgres 5433 / clickhouse 8124`
>    （容器间内部端口不变）。
> 2. **v4 必须开 dual write mode**：Langfuse v4 默认 `LANGFUSE_MIGRATION_V4_WRITE_MODE=events_only`，
>    会拒绝旧 SDK（如 langfuse 3.x Python SDK）的 trace 写入，事件只进 `events_core`
>    而不展开到 UI 读取的 `traces`/`observations` 表（worker 报 `No partitions available`）。
>    必须在 compose 的 `langfuse-worker-env` anchor 里加 `LANGFUSE_MIGRATION_V4_WRITE_MODE: dual`
>    并 `docker compose down && docker compose up -d` 重启才生效。
> 3. **客户端 SDK 契约**：langfuse 3.x 的 `CallbackHandler()` 为**无参构造**，
>    public_key/secret_key/host 全部从环境变量读取（见 `cognia/observability.py`）。
> 4. **预置 project**：本机 `.env` 用 `LANGFUSE_INIT_*` 预设了 org/project/keys
>    （project: `cognia-prod`），无需进 UI 手动建。

- Web UI：http://localhost:3001
- `.env` 填：
  ```
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_SECRET_KEY=sk-lf-...
  LANGFUSE_HOST=http://localhost:3001
  ```

## 验证

1. 启动服务，跑一段含「构建目标 → 诊断 → 讲解」的对话。
2. 打开 Langfuse → Tracing，应出现 1 条 trace：
   - 主 agent 每轮 ReAct teacher LLM 节点
   - 9 个工具节点 span
   - 工具**内部**子 LLM 节点（诊断 / 探针 / 讲解 / 知识建模）嵌套在对应工具节点下
3. 点开任意 generation：并排看 model / input+output tokens / cost / latency，
   prompt 与 output 分 tab 展示。

## 关闭

把 `.env` 三个 `LANGFUSE_*` 变量留空（或删掉）即可，无需改代码、无需重启之外的副作用。

## 排查

- `LANGFUSE_HOST` 必须带协议前缀（`http://` / `https://`），否则初始化失败（此时自动降级为关闭并告警）。
- 开启后仍看不到 trace：确认服务进程确实读到了 `.env`（load_dotenv 在 cognia/server.py 入口）。
- 诊断 502 降级时 trace 不丢：propose_diagnosis 的 try/except 降级路径仍正常记录（output 含 `recorded: false`）。

## 改动清单（本期）

- 新增 `cognia/observability.py`：全局 `LANGFUSE_HANDLER` 单例（三变量齐全才构建）。
- `cognia/server.py`：endpoint 注入 `config["callbacks"]` + 会话元数据（user_id / session_id）。
- `cognia/models.py` / `learning_engine.py` / `wiki_summarize.py` / `tools.py`：
  透传 `config` 到工具内部子 LLM 调用。
- `pyproject.toml`：新增 `langfuse` 依赖。
