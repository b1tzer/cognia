#!/usr/bin/env python3
"""一次性创建 GitHub issue 三层结构（需求→用例→任务）并建立 sub-issue 关联。

用法：uv run python scripts/create_wiki_issues.py
依赖：gh CLI（已登录 b1tzer，token 含 repo 权限）。
仅使用 Python 标准库，通过 subprocess 调用 gh api / gh api graphql。
"""

import json
import subprocess
import sys

REPO = "b1tzer/cognia"


def gh(args: list[str]) -> str:
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ERROR] gh {' '.join(args)}\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return r.stdout


def create_issue(title: str, body: str, labels: list[str]) -> dict:
    cmd = ["api", "--method", "POST", f"repos/{REPO}/issues",
           "-f", f"title={title}", "-f", f"body={body}"]
    for lab in labels:
        cmd += ["-f", f"labels[]={lab}"]
    out = gh(cmd)
    return json.loads(out)


def add_sub_issue(parent_node_id: str, child_node_id: str) -> None:
    query = ("mutation($p:ID!,$c:ID!){"
             "addSubIssue(input:{issueId:$p,subIssueId:$c}){clientMutationId}}")
    gh(["api", "graphql",
        "-f", f"query={query}",
        "-F", f"p={parent_node_id}",
        "-F", f"c={child_node_id}"])


def main() -> None:
    # ---- 需求 ----
    req_body = """## 背景
Cognia 第三资产：把学习对话「物化」为体系化、人可读、可在线编辑的知识长文（Personal Wiki）。
方案见 `docs/wiki-system-design.md`。

## 关键决策
1. **路线 B**：wiki 独立体系，不复用知识版图 DAG，映射暂不维护
2. **触发**：会话结束后总结；前端手动按钮为主 + Agent 结束回复提示
3. **版本+溯源**：Git（每 user 独立仓库）+ source_thread_id/turns/evidence/author
4. **AI 草稿 vs 用户手改**：用户手改 author=user，AI 只追加新版本不覆盖
5. **总结能力**：封装为 tool + `/wiki/summarize` 端点

## 用例清单
- UC1 会话结束总结为 wiki 草稿
- UC2 在线查看与编辑 wiki 页面
- UC3 wiki 版本历史与溯源

## Out of Scope
wiki 与 DAG 映射、全文搜索、多用户协作、权限、真实账号。

## DoD
3 个任务 PR 全部合并，正常/边界/异常 AC 全过，自测通过。"""
    req = create_issue(
        "[需求] 个人 Wiki：对话总结为体系化知识文本 + 在线查看编辑",
        req_body,
        ["kind/requirement", "status/todo", "priority/high",
         "type/feature", "area/backend", "area/frontend", "assignee/pm"],
    )

    # ---- 用例 ----
    uc1_body = """## 用户故事（INVEST）
作为学习者，我希望在一次学习会话结束后，一键把本次对话总结为一篇体系化的 wiki 文章，以便沉淀可复读的知识。

## 验收标准（AC）
**正常**
- 会话结束后点击「生成 wiki」，后端读该 thread 对话历史，LLM 提炼为 markdown，产出草稿页（author=ai），出现在 wiki 列表且可查看。
**边界**
- 空对话 / 无实质内容 → 返回「无可总结内容」提示，不产出空页面。
- 用户手改后（author=user）再次总结同一会话 → 追加新版本，不覆盖用户手改正文。
**异常**
- user_id 缺失 → 400；Postgres/Git 不可用 → 降级返回，不 500。"""
    uc1 = create_issue(
        "[用例] UC1 会话结束总结为 wiki 草稿",
        uc1_body,
        ["kind/use-case", "status/todo", "priority/high", "type/feature", "area/backend"],
    )

    uc2_body = """## 用户故事（INVEST）
作为学习者，我希望在浏览器里查看、编辑 wiki 页面（markdown），保存后立即生效，以便维护自己的知识库。

## 验收标准（AC）
**正常**
- 能列出全部 wiki 页面；点开查看渲染的 markdown；进入编辑模式修改并保存，内容更新且产生新版本。
**边界**
- 编辑保存后 author 标记为 user。
- 空正文保存 → 校验拒绝并提示。
**异常**
- 页面不存在 → 404；后端不可用 → 前端提示错误，不白屏。"""
    uc2 = create_issue(
        "[用例] UC2 在线查看与编辑 wiki 页面",
        uc2_body,
        ["kind/use-case", "status/todo", "priority/high", "type/ui", "area/frontend"],
    )

    uc3_body = """## 用户故事（INVEST）
作为学习者，我希望看到每篇 wiki 的版本历史与溯源信息（来自哪次会话、引用哪些原话），并能回滚到历史版本。

## 验收标准（AC）
**正常**
- 每篇页面能列出版本历史（commit + 时间 + 作者）。
- AI 生成的页面标注 source_thread_id + evidence（用户原话片段）。
- 回滚到指定历史版本后正文恢复。
**边界**
- 无历史（仅一版）→ 历史列表仅当前版本。
**异常**
- 不存在的 commit → 返回 400。"""
    uc3 = create_issue(
        "[用例] UC3 wiki 版本历史与溯源",
        uc3_body,
        ["kind/use-case", "status/todo", "priority/medium", "type/feature", "area/backend"],
    )

    # ---- 任务 ----
    t1_body = """## 契约
新增后端数据层：
- `cognia/wiki_repo.py`：Git 版本层（每 user 独立仓库，subprocess 调 git CLI，零新依赖）。提供 write/read/history/rollback。
- `cognia/wiki.py`：Store 元数据 + 领域逻辑（`("wiki", user_id)` namespace），CRUD + 溯源字段。
- `cognia/schemas.py`：新增 `WikiAuthor` / `WikiPage` / `WikiRevision`。
- `cognia/routers/wiki.py`：`/wiki` CRUD + history + rollback 路由，挂到 `server.py`。

## 任务级验收标准
- 不依赖 LLM，CRUD + 历史 + 回滚跑通；单元测试覆盖；store/Git 缺失安全降级不 500。"""
    t1 = create_issue(
        "[任务] wiki 后端数据层：Git 版本 + Store 元数据 + CRUD 路由",
        t1_body,
        ["kind/task", "status/todo", "priority/high", "type/feature", "area/backend", "assignee/pm"],
    )

    t2_body = """## 契约
新增后端总结层：
- summarize 核心函数（读 checkpointer 对话历史 → LLM 提炼 → 写 wiki 草稿）。
- `POST /wiki/summarize` 端点（前端按钮触发）。
- `summarize_session_to_wiki` Agent tool（异步，AI 可控调用），与端点共用核心函数。

## 任务级验收标准
- mock LLM 跑通「对话 → wiki 草稿」；空对话返回「无可总结」；溯源字段（source_thread_id/evidence/author=ai）正确落库。"""
    t2 = create_issue(
        "[任务] wiki 后端总结层：summarize 端点 + tool",
        t2_body,
        ["kind/task", "status/todo", "priority/high", "type/feature", "area/backend", "assignee/pm"],
    )

    t3_body = """## 契约
新增前端：
- 视图 `wiki`（与 chat/map 并列，header 入口）。
- 列表页 + 查看页（react-markdown 渲染）+ 编辑页（@uiw/react-md-editor）+ 历史/回滚面板。
- 「生成 wiki」按钮 → 调 `/wiki/summarize`。
- `frontend/app/lib/wiki.ts` + `frontend/app/api/wiki/[[...path]]/route.ts` 代理。

## 任务级验收标准
- 浏览器里能查看/编辑/保存/看历史/点按钮生成；样式沿用 design token，不硬编码颜色。"""
    t3 = create_issue(
        "[任务] wiki 前端：列表/查看/编辑页 + 生成按钮",
        t3_body,
        ["kind/task", "status/todo", "priority/high", "type/ui", "area/frontend", "assignee/pm"],
    )

    # ---- sub-issue 关联：需求→用例，用例→任务 ----
    add_sub_issue(req["node_id"], uc1["node_id"])
    add_sub_issue(req["node_id"], uc2["node_id"])
    add_sub_issue(req["node_id"], uc3["node_id"])
    add_sub_issue(uc1["node_id"], t2["node_id"])  # UC1 → 任务② 总结层
    add_sub_issue(uc2["node_id"], t3["node_id"])  # UC2 → 任务③ 前端
    add_sub_issue(uc3["node_id"], t1["node_id"])  # UC3 → 任务① 数据层

    print("=== 创建完成 ===")
    for name, obj in [("需求", req), ("UC1", uc1), ("UC2", uc2), ("UC3", uc3),
                      ("任务①", t1), ("任务②", t2), ("任务③", t3)]:
        print(f"{name}: #{obj['number']} {obj['html_url']}")


if __name__ == "__main__":
    main()
