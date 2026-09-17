# Cognia 前端审美规范（给 AI 派活时强制塞入）

> 本文是「约束」不是「科普」。派任何前端活时，把本文件整段贴进 prompt 的「## 约束」节。
> 判断依据：另一个大模型读完这一条，会因此多做一个正确动作或少做一个错误动作吗？不会的条目都砍掉。

## 1. 视觉语言（一句话）

低饱和中性底 + 墨色 + 单一强调色，DAG 点线星图，认知状态映射节点明暗/填充度。克制，不炫技。

## 2. 唯一颜色来源

- 所有颜色一律引用 `frontend/app/globals.css` 里的 token，禁止在组件内硬编码 hex 或 rgb。
- 换强调色只改 `globals.css` 的 `--accent` 一处，不许在组件里写第二个彩色。

| 语义 | token | Tailwind 类名 |
|------|-------|--------------|
| 页面底 | `--background` | `bg-background` |
| 卡片/面板 | `--surface` | `bg-surface` |
| 次级底/hover | `--surface-muted` | `bg-surface-muted` |
| 正文 | `--foreground` | `text-foreground` |
| 次要文字 | `--muted` | `text-muted` |
| 边框/分隔线 | `--line` | `border-line` |
| 强调（唯一彩色） | `--accent` | `bg-accent` / `text-accent` / `border-accent` |
| 强调 hover | `--accent-hover` | `bg-accent-hover` |

## 3. 强制规则（MUST / NEVER）

1. MUST 全站彩色只有 `--accent` 一个装饰色。按钮、链接、选中态、focus 环全部用 accent。
2. NEVER 用 Tailwind 默认彩色系（`violet-*` `blue-*` `emerald-*` `rose-*` `amber-*` `sky-*`）做装饰。这些只允许出现在「语义色」里（见第 4 节）。
3. MUST 中性色只用 token 映射（`bg-background` / `bg-surface` / `text-foreground` / `text-muted` / `border-line`），不要写 `bg-white` / `text-zinc-*`。
4. MUST 支持深浅两种模式：写样式时用 token，禁止写死 `bg-white`、`text-black`。
5. MUST 正文字号 `text-sm`（14px），标题 `text-base`（16px）→ `text-lg`（18px）→ `text-xl`（20px）三级，行高正文 `leading-relaxed`。

## 4. 语义色 vs 装饰色（关键区分）

- **语义色**（承载认知状态信息，允许彩色）：已掌握 `emerald`、部分掌握 `amber`、错误理解 `rose`、盲区 `sky`、未评估 `zinc`。这些只用于知识图谱节点/状态徽标，一个状态一个色，全站统一。
- **装饰色**（按钮/边框/链接/选中/分割，只允许一种）：`--accent`。

判据：这个颜色是不是在传达「认知状态」？是 → 语义色；否 → 只能用 accent 或中性色。

## 5. 间距 / 圆角体系（只用这几个，禁止乱造）

- 间距：`4 / 8 / 12 / 16 / 24 / 32`（对应 `p-1` `p-2` `p-3` `p-4` `p-6` `p-8`）。
- 圆角：小 `rounded`(4) / 卡片 `rounded-lg`(8) / 面板 `rounded-xl`(12)。
- 阴影：最多一档 `shadow-sm`，大面积留白不靠阴影撑，靠间距和边框分组。

## 6. 评审 checklist（改完逐条勾，全过才算完成）

1. 一眼先看到最重要的内容吗？（标题与正文层级分明）
2. 全站彩色超过 1 个吗？（超过 = 杂，砍）
3. 元素之间喘得过气吗？（挤 = 间距 < 8px 且无分组）
4. 同类元素在一条对齐线上吗？
5. 有无为炫技加的无意义动效/阴影？（有 = 删）

## 7. 现状迁移映射（存量代码怎么改）

当前三个文件还在用默认色板，迁移按此表替换，不新增彩色：

| 现状 | 改为 |
|------|------|
| `bg-white` | `bg-surface` |
| `bg-zinc-50` / `bg-zinc-100` | `bg-surface-muted` |
| `text-zinc-900` / `text-zinc-800` | `text-foreground` |
| `text-zinc-500` / `text-zinc-400` / `text-zinc-600` | `text-muted` |
| `border-zinc-200` / `border-zinc-300` | `border-line` |
| `bg-zinc-900`（发消息按钮/气泡） | `bg-accent` + `text-white` |
| `violet-*`（选择题交互）、`blue-*`（诊断面板）、`emerald-*`（地图面板）三种装饰彩色 | 统一收编为 `accent`；其中「已掌握/错误理解」等状态色保留语义色不动 |

## 8. 派活 prompt 模板（复制即用）

```
## 约束
- 视觉遵循 docs/frontend-aesthetics-guide.md，先读该文件。
- 颜色只用 globals.css 的 token，禁止硬编码 hex/rgb。
- 全站装饰彩色只有 --accent 一个；认知状态语义色（emerald/amber/rose/sky）只用于节点/徽标。
- 中性色禁止写 bg-white / text-zinc-*，一律用 bg-surface / text-foreground / text-muted / border-line。
- 正文字号 text-sm，间距只用 4/8/12/16/24/32，圆角只用 rounded/rounded-lg/rounded-xl。
- 改完跑一遍第 6 节 checklist，5 条全过再交。
```

## 9. 自检信号（完成标准）

- `grep -rn "bg-white\|text-zinc\|bg-zinc\|border-zinc\|violet-\|blue-200\|emerald-200" frontend/app` 结果为空（语义色除外）。
- 全站 hex 色值只出现在 `globals.css` 和 `knowledge-map.tsx` 的认知状态节点色里，其余地方为零。
- 深浅两种模式各截一张图，页面不塌、层级分明、彩色只有 accent + 语义色。
