# Cognia 项目操作流程 SOP（Standard Operating Procedures）

> 本文件集中记录 Cognia 项目日常协作与开发的**标准作业程序**，供 PM 与团队成员对齐执行。
> 每一条 SOP 都以「何时用 → 怎么做 → 验收标准」三段式描述，确保可落地、可校验。
>
> **层级说明**：所有需求的**通用交付元流程**（调研 → 设计 → 开发 → 验收）见 [docs/general-sop.md](./general-sop.md)；本文件是它在本项目下的**具体场景补充**（issue 协作、前后端分工、派活规则等）。

---

## 1. AAR 事后复盘 SOP

**何时用**：每个 Phase / 里程碑完成后必须执行一次；重大偏差（成功或失败）发生后 24-48 小时内执行。

**怎么做**：按顺序回答四个问题，逐条作答，不跳步、不追责：

1. **当初的意图是什么？**（What was supposed to happen?）—— 复述计划、目标、成功标准。
2. **实际发生了什么？**（What actually happened?）—— 只陈述客观事实，不评判好坏。
3. **为什么会有差异？**（Why was there a difference?）—— 追系统原因，不追个人过错。
4. **下次如何做得不同？**（What will we do differently next time?）—— 产出具体可执行的行动。

**产出物格式**（三部分）：

- 四个问题逐一作答
- **保持项 Sustains**：已验证有效、应固化的做法
- **优化项 Improve**：每条必须具体、可执行、带 owner 和时机

**验收标准**：
- 每条结论都落到「可执行的下一步行动」（含 owner + 时机），而非空泛口号
- 优化项中「跨任务可复用的工作流程纪律」已沉淀为长期记忆
- 项目级规范已同步到本 SOP 文档

---

## 2. GitHub issue 任务协作 SOP

**何时用**：任何需要多人协作或需要记录上下文的任务，统一走 GitHub issue（仓库 `b1tzer/cognia`）。

**怎么做**：

1. **方案先行**：开发者在 issue 评论区先给出方案要点（改什么、怎么改、验收标准）。
2. **PM 确认**：PM 回复确认后才允许动工。
3. **切分支**：从 `main` 切 `fix/*` 分支开发（切完立即 `git branch --show-current` 验证）。
4. **提交 PR**：描述写「改动 + 修复 + 验收结果」。
5. **Code review**：PM 在 PR 评论做 review，通过则合并，未通过按评论修订。**注意**：PM 与牛马 1 号共用 `b1tzer` GitHub 账号，GitHub 会拒绝「approve 自己的 PR」（`gh pr review --approve` 报错 `Cannot approve your own pull request`），故 review 结论一律用 `gh pr review --comment`（而非 `--approve`），合并用 `gh pr merge` 直接执行。
6. **验收关闭**：PM 在 issue 评论验收结论，关闭 issue。

**标签体系**：

| 维度 | 标签 |
| --- | --- |
| 状态 | `status/todo` `status/in-progress` `status/review` `status/done` `status/blocked` |
| 优先级 | `priority/urgent` `priority/high` `priority/medium` `priority/low` |
| 类型 | `type/feature` `type/ui` `type/refactor` `type/bugfix` |
| 层次 | `kind/requirement` `kind/use-case` `kind/task`（需求/用例/任务三层） |
| 领域 | `area/frontend` `area/backend` |
| 人员 | `assignee/pm` `assignee/牛马1号`（agent 无 GitHub 账号，用标签标记人） |

**验收标准**：
- 每个任务都带「需求基线 + 验收标准」
- 状态标签随进度流转（todo → in-progress → review → done）
- PR 描述含「改动 / 修复 / 验收结果」三段

---

## 3. 后端开发 SOP（PM 负责）

**何时用**：所有后端核心紧耦合开发由 PM 本人完成。

**怎么做**：

1. 遵循「确定性骨架 + LLM 语义决策」混合架构（见 `backend/decision.py`）。
2. 遵守分层 AI 原则：诊断层重（北极星指标载体）、决策层轻、认知更新层零 token。
3. 层间通信用结构化 JSON 而非自然语言摘要。
4. **每改一处配一处测试**，用 `git commit` 做版本锚点。
5. 改完跑全量回归 + lint。

**验收标准**：
- 新增代码有对应单测覆盖
- 全量回归全绿（`python -m unittest discover -s . -p "test_*.py"`）
- lint 无告警
- 行为等价（不破坏现有降级逻辑）

---

## 4. 前端开发 SOP（牛马 1 号负责）

**何时用**：所有前端工作（含改一行 CSS）都由「牛马 1 号」Agent 完成，PM 绝不自己动手。

**怎么做**：

1. PM 提 GitHub issue，定义契约与验收标准。
2. 牛马 1 号走 issue → `fix/*` 分支 → PR 流程。
3. PM 在 PR 上做 code review，结果校验。

**CSS 最小改动红线**：每次编辑前先明确「唯一要改的属性清单」，其余属性逐字原样保留；改完用 `git diff` 逐行核对，确认没碰清单之外的属性。

**验收标准**：
- 前端改动全部走 issue → PR → review → 合并流程
- CSS 改动满足「最小改动」红线（不误删无关属性）

---

## 5. 派活决策 SOP

**何时用**：接到新任务时，判断自己做还是派发给其他 agent。

**怎么做**：

- **自己做**：紧耦合迭代型任务（改代码 + 跑测试等反馈回路极短的核心工作）。
- **派发**：独立模块开发、信息收集/调研、造数据、文档编写等可并行任务，优先派发。

**验收标准**：
- 可并行、可独立、边界清晰的任务已派发
- PM 只做「契约定义 + 结果校验」，不代劳可并行任务

---

## 6. 测试与验收 SOP

**何时用**：每次代码改动完成后。

**怎么做**：

1. 单测：`cd backend && .venv/bin/python -m unittest test_xxx -v`。
2. 全量回归：`cd backend && .venv/bin/python -m unittest discover -s . -p "test_*.py"`（依赖装在 `backend/.venv`，系统 Python 会缺 `fastapi` 等依赖，勿用系统 Python）。
3. lint 检查。
4. 提交 PR → review → squash 合并。
5. issue 验收评论 → 关闭。

**验收标准**：
- 单测通过
- 全量回归全绿
- lint 无告警
- issue 带验收结论后关闭

---

## 7. 需求/用例/任务三层 issue 结构 SOP

**何时用**：当一个需求可拆解为「多个用例、每个用例下又有可执行任务」时，用三层 issue 结构记录，确保全过程文档化、不丢失内容。

**怎么做**：

1. **三层结构**：需求（Requirement，1 个）→ 用例（Use Case，N 个）→ 任务（Task，M 个），逐层用 GitHub 原生 Sub-issue 建立父子关联。
2. **标题前缀区分层次**：`[需求]` / `[用例]` / `[任务]` 打头，一眼可辨层次。
3. **标签区分层次**：`kind/requirement`（需求）/ `kind/use-case`（用例）/ `kind/task`（任务），与标题前缀双保险。
4. **内容分工**：
   - 需求 issue：背景 + 方案概述 + 用例清单 + 关键决策 + Out of Scope + DoD。
   - 用例 issue：用户故事（INVEST）+ 验收标准 AC（覆盖正常/边界/异常）。
   - 任务 issue：契约（做什么、涉及文件/接口）+ 任务级验收标准。
5. **关联建立**：`gh issue create` 本身不支持 `--sub-issue` 参数，需用 GraphQL `addSubIssue` mutation（`gh api graphql`）挂父子关系。

**验收标准**：
- 三层结构完整：需求下有全部用例、每个用例下有对应任务，无遗漏
- 每个 issue 的标题前缀 + 标签与所属层次一致
- Sub-issue 树可完整追溯（需求 → 用例 → 任务）
- 全过程（背景/方案/AC/契约）均有文档记载，不丢失内容

---

## 8. 派活 prompt 编写规范 SOP

**何时用**：PM 通过 DirectTalk（`imate agent invoke`）或其他方式给牛马 1 号等 agent 派活、写任务指令时，必须遵守本规范，确保 prompt 清晰、可执行、agent 能准确理解并正确收尾。

**背景依据**（多方交叉验证：AWS/Oracle/Anthropic/JetBrains 等 agent prompt 工程最佳实践）：agent 失败大多不是模型问题，而是 prompt 问题——模糊指令 → 错误动作；缺完成标准 → 提前宣布完成；缺收尾步骤 → 成果不可见。

**怎么做**（按以下固定结构组织派活 prompt）：

1. **分节标题**：用 `## 身份 / ## 任务 / ## 约束 / ## 完成标准 / ## 禁止事项 / ## 示例` 等 Markdown 标题分节，LLM 对结构化内容解析更可靠，忌大段无结构散文。
2. **角色先行**：第一句给定角色（如「你是 Cognia 前端开发牛马 1 号」），锚定视角与决策标准。
3. **约束优先**：先说 **NOT to do**（禁止项），再说 to do（要做项）。用「约束 + 范围 + 验证」三件套：只改哪些文件、绝不碰哪些、改完怎么验证。
4. **完成标准可观察**：用 3-5 条**具体可观察**的完成标准（AC），而非「做好/优化/改进」这类模糊词。**「定义类型 ≠ 完成功能」**——完成标准必须落在**可见行为**上（如「桥接节点被高亮并显示文案」「点击节点出现按钮」），不能是「定义了某 interface」。
5. **一条一个行为**：每个 bullet 只放一个原子指令，不用 `A 且 B 或 C` 的复合句。
6. **具体动词**：用「改/加/删/调/跑」等动作词，避免「explore/improve/clean up」这类不可验证词。
7. **明确收尾操作**（关键，agent 常漏）：每个任务末尾显式列出收尾步骤——`git add → git commit → git push -u origin <branch> → gh pr create`，并**要求回报 PR 链接**。宁可每个子任务完成后立即 push+PR，也不要攒到最后（防止超时导致成果丢失）。
8. **引用现有样例**：让 agent「follow 现有文件/issue 的模式」，代码库就是最好的风格指南，胜过用文字描述风格。
9. **强指令词**：只在**高风险**约束处用大写 MUST/MUST NOT/NEVER（如目录隔离、红线），其余用小写，避免「全是大写等于没强调」。
10. **失败状态兜底**：明确「接口不确定时在 issue 提问、勿臆造」「超时/报错时先回报已完成的 PR 再继续」。

**验收标准**：
- 派活 prompt 含「角色 + 约束(先 NOT 后 DO) + 可观察完成标准 + 明确收尾步骤(push+PR+回报链接)」
- 完成标准是可见行为，不含「定义类型=完成功能」这类误判
- agent 收到后能独立完成到「push + PR + 回报链接」闭环，无需 PM 追问收尾

---

**附：牛马 1 号历史教训（反例，派活时对照自查）**

| 踩过的坑 | 对应规范条款 |
| --- | --- |
| 定义了 `AtlasBridge` 类型却没渲染 | 条款 4：完成标准必须落在可见行为 |
| 写了 #39/#40 代码但没 push、没提 PR | 条款 7：明确收尾操作 + 小步立即 push |
| 分支串行叠加（fix/37 基于 fix/36） | 条款 7/约束：每个 issue 从 main 独立切分支 |
| `agent invoke` 30 分钟同步阻塞超时被取消 | 条款 7：小步产出、每个 issue 立即 push+PR |

---

## 9. 判断 agent 对话/任务是否完成 SOP

**何时用**：给牛马 1 号派活后，`imate agent invoke` 是同步阻塞命令，会「假死」（超时不返回、被取消），此时**不能靠 imate 返回判断任务是否完成**，必须直接查对话记录数据库。

**关键区分**：「对话是否完成」（agent 是否已给出最终答复）≠「工作是否有产出」（代码/PR 有没有）。前者查对话记录 DB，后者查 git/gh。两者不能混。

**最权威做法（唯一不用猜的方式）：直接查对话记录数据库**

牛马 1 号跑在 hermes，它的对话记录存在本地 SQLite 数据库 **`~/.hermes/state.db`**，完成与否有**明确的标识字段**，直接查字段值，不猜：

**① 消息级完成标识 —— `messages` 表的 `finish_reason` 字段**

```
sqlite3 ~/.hermes/state.db \
  "SELECT id, role, finish_reason, substr(content,1,80), datetime(timestamp,'unixepoch','localtime')
   FROM messages WHERE session_id='<session_id>' ORDER BY id DESC LIMIT 10;"
```

- `finish_reason` 取值含义：`stop` = 正常结束、已生成最终答复；`tool_calls` = 还在调工具、未结束；`NULL` = 工具执行结果。
- **判断「对话是否说完最终答案」**：看该 session 最后一条 `assistant` 消息的 `finish_reason` 是否为 `stop`。

**② 会话级结束标识 —— `sessions` 表的 `ended_at` + `end_reason` 字段**

```
sqlite3 ~/.hermes/state.db \
  "SELECT id, datetime(started_at,'unixepoch','localtime'), datetime(ended_at,'unixepoch','localtime'), end_reason, message_count
   FROM sessions ORDER BY started_at DESC LIMIT 20;"
```

- `ended_at` 非空 + `end_reason` 有值 = 会话已正式关闭。
- `end_reason` 取值：`compression`(上下文压缩切分) / `ws_orphan_reap`(websocket 孤儿回收) / `cron_complete`(定时任务完成) / `session_reset`(重置) / `NULL`(未结束)。

**③ 关键结论：两个标识可能不一致，这是假死/超时的典型特征**

imate invoke 同步阻塞 + 超时取消，会导致「agent 已说完最终答案（`finish_reason=stop`），但会话未被正式关闭（`ended_at`/`end_reason` 仍为 `NULL`）」。**这不代表任务没做完**——任务是否做完，以 `finish_reason=stop` 且最后一条 assistant 消息内容为准。

**判断规则**：
1. 先查 `messages` 表最后一条 assistant 消息的 `finish_reason`：`stop` = 已给出最终答复；`tool_calls` = 还在干。
2. 再查 `sessions` 表的 `ended_at`/`end_reason`：非空 = 会话已关闭；`NULL` = 会话未关闭（可能是 invoke 超时取消导致）。
3. 会话 id 来源：从 `sessions` 表按 `started_at` 倒序取最新；或从派活返回的 `session_id` 字段（imate 结果 JSON 里有）。

**⚠️ 反例（历次实测验证，勿再犯）**：
- `~/.hermes/gateway_state.json` 的 `active_agents` 字段**不可靠**（mtime 滞后 2 天，非实时），不能作为判断依据。
- `~/.hermes/logs/agent.log` 尾部是 `model-probe`/`model_refresh` 时，那是后台心跳，不代表有实际任务。
- `git/gh` 产出是「结果」，不是「对话是否完成」的判断依据。

**验收标准**：
- 能说出「牛马 1 号最后一次对话是否已给出最终答复」（查 messages 表 finish_reason）
- 能说出「该会话是否已正式关闭、若未关闭是什么原因」（查 sessions 表 ended_at/end_reason）
- 判断依据来自 `~/.hermes/state.db` 的字段值，不靠日志/进程/网关状态猜测
