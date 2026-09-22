# Cognia 个人 Wiki 子系统设计（Personal Wiki）

> 状态：已拍板，进入开发。
> 定位：Cognia 的第三资产——把学习对话「物化」为体系化、人可读、可在线编辑的知识长文。
> 与既有两套知识载体并列：① 对话记忆（Checkpointer，过程）；② 知识版图（Knowledge Model + Proficiency，机器可读认知状态）；③ 本文档定义的 Personal Wiki（人可读知识文本）。

## 0. 已拍板决策

| # | 决策点 | 结论 |
|---|--------|------|
| 1 | wiki 与知识版图的关系 | **路线 B**：独立体系，wiki 有自己的页面树，不复用 DAG 依赖边；映射关系暂不维护 |
| 2 | 自动总结触发时机 | 会话结束后总结；**前端手动按钮为主** + Agent 会话结束回复里提示「要不要把这次学习总结成 wiki」 |
| 3 | 溯源 + 版本 | 版本用 **Git**（每 user 独立仓库）；溯源记录 `source_thread_id + turns + evidence + author` |
| 4 | AI 草稿 vs 用户手改 | 用户手改标记 `author=user`；AI 后续总结**只追加新版本**、不覆盖用户手改内容 |
| 5 | 总结能力封装 | 核心逻辑封装为可复用函数 + Agent tool `summarize_session_to_wiki`（AI 可控调用）+ 独立端点 `/wiki/summarize`（前端按钮触发） |

## 1. 架构与存储分工

```
前端 Next.js                     后端 FastAPI + LangGraph
┌──────────────┐                 ┌────────────────────────────┐
│ wiki 列表/查看/编辑页 │◄────────►│ /wiki CRUD + /wiki/summarize │
│ @uiw/react-md-editor │         │ summarize_session_to_wiki     │
│ 「生成 wiki」按钮    │         │   (Agent tool, AI 可控调用)   │
└──────────────┘                 └──────┬──────────────┬────────┘
                                        │              │
                                   Git 仓库          Postgres Store
                             data/wiki/{user_id}/   ("wiki", user_id)
                             正文版本真相(diff/history)  元数据+溯源(结构化查询)
```

**分工原则**：
- **Git 存正文**：`data/wiki/{user_id}/` 每个 user 一个 git 仓库；正文文件 `{page_id}.md`。git commit = 版本，`git log` = 历史，`git diff` = 溯源差异。零新依赖（subprocess 调系统 git CLI）。
- **Postgres Store 存元数据**：`("wiki", user_id)` namespace，key = page_id，value = 元数据 dict（含 `commit_hash`、`author`、`source_thread_id`、`turns`、`evidence`、`title`、`tags`、`parent_page_id`）。溯源查询、页面列表、tags 过滤走这里。
- 正文与元数据的 commit_hash 对齐：每次写正文拿到 commit_hash 后，元数据同步记录该 hash。

## 2. 数据模型（schemas.py 新增）

```python
class WikiAuthor(str, Enum):
    AI = "ai"        # AI 生成的草稿
    USER = "user"    # 用户手改（不被 AI 覆盖）

class WikiPage(BaseModel):
    page_id: str                    # 稳定 id（slug，如 "spring-aop-proxy"）
    title: str
    content_markdown: str           # 正文（存 Git 文件）
    path: str = ""                  # 页面树路径（路线 B 独立体系，如 "java/spring/"）
    tags: list[str] = []
    parent_page_id: str | None = None
    author: WikiAuthor              # 最近一次写入者
    source_thread_id: str | None = None   # AI 生成时来源会话
    source_turns: list[int] = []          # 来源轮次（1-based）
    evidence: list[str] = []              # 溯源证据（用户原话片段，不脑补）
    updated_at: datetime = ...

class WikiRevision(BaseModel):
    page_id: str
    commit_hash: str
    author: WikiAuthor
    summary: str = ""               # 变更说明
    timestamp: datetime = ...
```

## 3. 接口契约

### 3.1 后端路由（routers/wiki.py，挂到 server.py）

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/wiki?user_id=` | 列出全部页面元数据（不含正文，按 updated_at 降序） |
| GET | `/wiki/{page_id}?user_id=` | 读单页（正文 + 元数据） |
| POST | `/wiki` | 新建页（body: title/content_markdown/tags/parent_page_id/author） |
| PUT | `/wiki/{page_id}` | 更新页（写 Git + 记版本 + 更新元数据） |
| GET | `/wiki/{page_id}/history?user_id=` | 读版本历史（git log） |
| POST | `/wiki/{page_id}/rollback` | 回滚到指定 commit（body: commit_hash） |
| POST | `/wiki/summarize` | 总结会话为 wiki 草稿（body: thread_id/title?） |

- 所有端点 `user_id` 缺失 → 400。
- store / Git 不可用 → 安全降级（列表返回 `[]`、写返回错误，不 500，不静默）。

### 3.2 Agent tool（tools.py 新增）

`summarize_session_to_wiki(title: str, config) -> str`：异步 tool，读当前 thread 对话历史（checkpointer）→ LLM 提炼 → 写 wiki 草稿（author=ai + source_thread_id + evidence）。与 `/wiki/summarize` 端点共用核心函数。

### 3.3 前端

- 新增视图 `wiki`（与 chat/map 并列），入口在 header。
- 列表页 + 查看页（react-markdown 渲染）+ 编辑页（@uiw/react-md-editor）+ 历史/回滚面板。
- 「生成 wiki」按钮：会话结束后点击 → 调 `/wiki/summarize`。
- 数据访问层 `frontend/app/lib/wiki.ts` + 代理 `frontend/app/api/wiki/[[...path]]/route.ts`。

## 4. 任务拆分（3 个 PR，链式依赖，合并用 --merge）

| 任务 | 交付 | 独立验证 |
|------|------|----------|
| ① 后端数据层 | `cognia/wiki_repo.py`(Git 版本层) + `cognia/wiki.py`(Store 元数据+领域逻辑) + schema + `cognia/routers/wiki.py` + 测试 | 不依赖 LLM，CRUD + 历史 + 回滚跑通 |
| ② 后端总结层 | summarize 核心函数 + `/wiki/summarize` 端点 + `summarize_session_to_wiki` tool + 测试 | mock LLM 跑通「对话 → wiki 草稿」 |
| ③ 前端 | wiki 列表/查看/编辑页 + 编辑器 + 生成按钮 + API 代理 | 浏览器里查看/编辑/保存/看历史/点按钮生成 |

## 5. 验收标准（AC）

**正常路径**：
- 会话结束后点「生成 wiki」→ 后端读对话历史 → LLM 提炼 → 产出 markdown 草稿 → 出现在 wiki 列表，可查看。
- 用户编辑正文保存 → 产生新 git commit → 历史面板出现该版本。

**边界**：
- 用户手改后（author=user），再次总结同一会话 → 追加新版本，不覆盖用户手改正文。
- 空对话（无实质内容）总结 → 返回「无可总结内容」提示，不产出空页面。

**异常**：
- user_id 缺失 → 400；Postgres / Git 不可用 → 降级返回，不 500。

## 6. Out of Scope（本次不做）

- wiki 与知识版图 DAG 的映射/联动（路线 B 明确暂缓）
- 全文语义搜索、多用户协作、权限体系、真实账号
- 页面间的自动关联推荐、标签自动聚类
