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
5. **Code review**：PM 在 PR 评论做 review，通过则合并，未通过按评论修订。
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
