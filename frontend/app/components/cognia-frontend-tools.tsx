"use client";

import { useState } from "react";
import {
  defineToolCallRenderer,
  useFrontendTool,
} from "@copilotkit/react-core/v2";
import { Markdown } from "@copilotkit/react-ui";
import { z } from "zod";

/**
 * Cognia 前端能力封装（CopilotKit 能力「三 + 五」）：
 *
 * 1. 前端工具（三）：useFrontendTool 注册 `practice_choice`，让后端 Cognia
 *    Agent 在诊断/干预后调用它，把「选择题练习」渲染成前端可交互卡片，
 *    而不是只能输出纯文本 Markdown。学生作答后前端本地判对错并展示解析（非阻塞、
 *    不主动触发 Agent 回答）；作答结果会暂存，待学生下次发言时作为一条
 *    「练习作答记录」消息带回给 Agent 继续推理。
 * 2. Generative UI（五）：defineToolCallRenderer 为 Cognia 已有后端工具
 *    （propose_diagnosis / build_learning_goal）注册自定义渲染器，把原本
 *    以 JSON 文本回流的工具结果渲染成结构化卡片。
 *
 * 这两类渲染器最终都由 useRenderToolCall()（在 page.tsx 中）统一驱动，
 * 使 headless 自定义渲染层也能吃到 CopilotKit 的 Generative UI 能力。
 */

// ---------------------------------------------------------------------------
// 前端交互工具：practice_choice（三）
// ---------------------------------------------------------------------------

type PracticeChoiceArgs = {
  question?: string;
  options?: string[];
  point_name?: string;
  correct_index?: number;
  explanation?: string;
};

/** 一次练习作答的结果，暂存后待用户下次发言时作为记录带回给 Agent。 */
export type PracticeAnswerRecord = {
  question: string;
  options: string[];
  point_name: string;
  selected_index: number;
  correct_index: number | null;
  is_correct: boolean;
};

// 模块级暂存：只保留「最近一次」作答结果（非阻塞方案，不在本轮触发 Agent）。
let pendingPracticeAnswer: PracticeAnswerRecord | null = null;

/** 取走并清空最近一次练习作答结果，供发送消息时带回 AI。 */
export function consumePendingPracticeAnswer(): PracticeAnswerRecord | null {
  const rec = pendingPracticeAnswer;
  pendingPracticeAnswer = null;
  return rec;
}

type PracticeChoiceRenderProps = {
  args: PracticeChoiceArgs;
  status: "inProgress" | "executing" | "complete";
};

function PracticeChoiceCard({ args, status }: PracticeChoiceRenderProps) {
  const [selected, setSelected] = useState<number | null>(null);

  const question = args?.question ?? "";
  const options = Array.isArray(args?.options) ? args.options : [];
  const pointName = args?.point_name ?? "";
  const rawCorrectIndex =
    typeof args?.correct_index === "number" ? args.correct_index : null;
  const correctIndex =
    rawCorrectIndex !== null && rawCorrectIndex >= 0 && rawCorrectIndex < options.length
      ? rawCorrectIndex
      : null;
  const explanation = args?.explanation ?? "";

  const answered = selected !== null;
  const isCorrect = answered && correctIndex !== null && selected === correctIndex;

  // 参数还在流式生成中，先给占位，避免渲染残缺内容。
  if (status === "inProgress" && !question && options.length === 0) {
    return (
      <div className="my-2 animate-pulse rounded-xl border border-zinc-200 bg-zinc-50 px-4 py-3 text-xs text-zinc-400">
        正在准备练习卡片…
      </div>
    );
  }

  const handleSelect = (i: number) => {
    if (answered) return;
    setSelected(i);
    const correct = correctIndex !== null && i === correctIndex;
    // 非阻塞：不主动触发 Agent 回答，只把作答结果暂存，待下次发言时带回。
    pendingPracticeAnswer = {
      question,
      options,
      point_name: pointName,
      selected_index: i,
      correct_index: correctIndex,
      is_correct: correct,
    };
  };

  return (
    <div className="my-2 overflow-hidden rounded-xl border border-violet-200 bg-violet-50/50">
      <div className="flex items-center gap-2 border-b border-violet-200 bg-violet-100/60 px-4 py-2">
        <span className="text-sm">📝</span>
        <span className="text-xs font-medium text-violet-700">
          交互练习{pointName ? ` · ${pointName}` : ""}
        </span>
      </div>
      <div className="px-4 py-3">
        {question && (
          <p className="mb-3 text-sm font-medium text-zinc-800">{question}</p>
        )}
        <div className="space-y-2">
          {options.map((opt, i) => {
            const isSelected = selected === i;
            let optionClass =
              "border-zinc-200 bg-white text-zinc-700 hover:border-violet-300";
            if (answered) {
              if (correctIndex !== null && i === correctIndex) {
                optionClass = "border-emerald-500 bg-emerald-50 text-emerald-900";
              } else if (isSelected) {
                optionClass = "border-rose-500 bg-rose-50 text-rose-900";
              } else {
                optionClass = "border-zinc-200 bg-white text-zinc-400";
              }
            } else if (isSelected) {
              optionClass = "border-violet-500 bg-violet-100 text-violet-900";
            }
            return (
              <button
                key={i}
                type="button"
                onClick={() => handleSelect(i)}
                disabled={answered}
                className={`flex w-full items-start gap-2 rounded-lg border px-3 py-2 text-left text-sm transition ${optionClass}`}
              >
                <span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-zinc-100 text-xs font-semibold text-zinc-600">
                  {String.fromCharCode(65 + i)}
                </span>
                <span className="flex-1 whitespace-pre-wrap">{opt}</span>
                {answered && correctIndex !== null && i === correctIndex && (
                  <span className="text-emerald-600">✓</span>
                )}
                {answered && isSelected && correctIndex !== i && (
                  <span className="text-rose-600">✗</span>
                )}
              </button>
            );
          })}
        </div>
        {answered && (
          <p
            className={`mt-3 text-sm font-medium ${
              isCorrect ? "text-emerald-700" : "text-rose-700"
            }`}
          >
            {isCorrect ? "✓ 回答正确" : "✗ 回答错误"}
          </p>
        )}
        {answered && explanation && (
          <div className="mt-2 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs text-zinc-600">
            <span className="font-medium text-zinc-700">解析：</span>
            <span className="whitespace-pre-wrap">{explanation}</span>
          </div>
        )}
      </div>
    </div>
  );
}

/** 注册 Cognia 的前端交互工具（三）。必须在 <CopilotKit> 内部调用。 */
export function useCogniaFrontendTools() {
  useFrontendTool({
    name: "practice_choice",
    description:
      "向学生展示一道交互式选择题（单选），用于巩固知识点、检验理解。" +
      "学生在前端点击作答后，前端会立即本地判对错并展示解析，不会阻塞你、也不会" +
      "主动触发你回答；作答结果会在学生下次发言时以「练习作答记录」的形式出现在" +
      "对话里，你届时据此推理即可。参数 correct_index 为正确选项的索引（从 0 开始），" +
      "explanation 为作答后展示的解析说明（含正确原因）。",
    parameters: z.object({
      question: z.string().describe("选择题题干，简洁明确"),
      options: z.array(z.string()).describe("2-4 个互斥选项"),
      point_name: z.string().describe("关联的知识点名称"),
      correct_index: z.number().describe("正确选项的索引（从 0 开始）"),
      explanation: z.string().describe("作答后展示的解析说明，解释为什么正确/错误"),
    }),
    render: PracticeChoiceCard,
  });
}

// ---------------------------------------------------------------------------
// Generative UI 渲染器：为后端工具渲染结构化卡片（五）
// ---------------------------------------------------------------------------

/** 认知状态五态 → 中文标签与颜色。 */
const STATE_META: Record<string, { label: string; color: string }> = {
  unassessed: { label: "未评估", color: "text-zinc-500 bg-zinc-100" },
  mastered: { label: "已掌握", color: "text-emerald-700 bg-emerald-100" },
  partial: { label: "部分掌握", color: "text-amber-700 bg-amber-100" },
  misconception: { label: "错误理解", color: "text-rose-700 bg-rose-100" },
  unknown: { label: "盲区", color: "text-sky-700 bg-sky-100" },
};

/** 解析后端工具的 JSON 字符串结果，失败时返回 null。 */
function parseToolResult(result: unknown): Record<string, unknown> | null {
  if (typeof result !== "string" || !result.trim()) return null;
  try {
    const parsed = JSON.parse(result);
    return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

/** propose_diagnosis 结果 → 认知诊断卡片。 */
const diagnosisRenderer = defineToolCallRenderer({
  name: "propose_diagnosis",
  args: z.object({
    point_id: z.string().optional(),
    point_name: z.string().optional(),
    point_description: z.string().optional(),
    question: z.string().optional(),
    user_answer: z.string().optional(),
    current_state: z.string().optional(),
  }),
  render: ({ args, status, result }) => {
    if (status !== "complete") {
      return (
        <div className="my-2 animate-pulse rounded-xl border border-zinc-200 bg-zinc-50 px-4 py-3 text-xs text-zinc-400">
          正在诊断…
        </div>
      );
    }

    const data = parseToolResult(result);
    if (!data) return null;

    const diagnosed = String(data.diagnosed_state ?? "");
    const confidence = String(data.confidence ?? "");
    const migrated = Boolean(data.migrated);
    const finalState = String(data.final_state ?? "");
    const evidence = Array.isArray(data.evidence) ? (data.evidence as string[]) : [];
    const meta = STATE_META[diagnosed] ?? { label: diagnosed || "未知", color: "text-zinc-500 bg-zinc-100" };

    return (
      <div className="my-2 overflow-hidden rounded-xl border border-blue-200 bg-blue-50/50">
        <div className="flex items-center gap-2 border-b border-blue-200 bg-blue-100/60 px-4 py-2">
          <span className="text-sm">🧠</span>
          <span className="text-xs font-medium text-blue-700">
            认知诊断 · {args.point_name ?? args.point_id ?? ""}
          </span>
        </div>
        <div className="space-y-2 px-4 py-3 text-sm">
          <div className="flex items-center gap-2">
            <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${meta.color}`}>
              {meta.label}
            </span>
            <span className="text-xs text-zinc-500">置信度 {confidence}</span>
            {migrated && (
              <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
                状态已迁移 → {STATE_META[finalState]?.label ?? finalState}
              </span>
            )}
          </div>
          {evidence.length > 0 && (
            <div className="text-xs text-zinc-500">
              <span className="font-medium text-zinc-600">依据：</span>
              {evidence.map((e, i) => (
                <span key={i} className="mr-2 inline-block rounded bg-white px-1.5 py-0.5">
                  “{e}”
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  },
});

/** build_learning_goal 结果 → 知识模型卡片。 */
const knowledgeModelRenderer = defineToolCallRenderer({
  name: "build_learning_goal",
  args: z.object({ goal: z.string().optional() }),
  render: ({ status, result }) => {
    if (status !== "complete") {
      return (
        <div className="my-2 animate-pulse rounded-xl border border-zinc-200 bg-zinc-50 px-4 py-3 text-xs text-zinc-400">
          正在构建知识模型…
        </div>
      );
    }

    const data = parseToolResult(result);
    if (!data) return null;

    const action = String(data.action ?? "");
    const goal = String(data.goal ?? "");
    const count = Number(data.point_count ?? 0);
    const points = Array.isArray(data.points) ? (data.points as string[]) : [];

    return (
      <div className="my-2 overflow-hidden rounded-xl border border-emerald-200 bg-emerald-50/50">
        <div className="flex items-center gap-2 border-b border-emerald-200 bg-emerald-100/60 px-4 py-2">
          <span className="text-sm">🗺️</span>
          <span className="text-xs font-medium text-emerald-700">
            知识模型{action ? ` · ${action}` : ""}
          </span>
        </div>
        <div className="px-4 py-3 text-sm">
          <p className="mb-2 font-medium text-zinc-800">{goal}</p>
          <p className="mb-2 text-xs text-zinc-500">共 {count} 个知识点</p>
          <ol className="list-inside list-decimal space-y-1 text-xs text-zinc-600">
            {points.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ol>
        </div>
      </div>
    );
  },
});

/** explain 结果 → 讲解正文卡片。

 * 讲解正文是「数据」（markdown 文本），必须正常渲染展示，而不是被折叠成
 * 「🔧 调用完成」灰字。这里直接用 <Markdown> 渲染 tool 返回的讲解全文，
 * 保证「先讲后练」的讲解内容能被用户看到。
 */
const explainRenderer = defineToolCallRenderer({
  name: "explain",
  args: z.object({
    point_name: z.string().optional(),
    point_description: z.string().optional(),
    user_state: z.string().optional(),
  }),
  render: ({ args, status, result }) => {
    if (status !== "complete") {
      return (
        <div className="my-2 animate-pulse rounded-xl border border-zinc-200 bg-zinc-50 px-4 py-3 text-xs text-zinc-400">
          正在讲解…
        </div>
      );
    }

    const text = typeof result === "string" ? result : "";
    if (!text.trim()) return null;

    return (
      <div className="my-2 overflow-hidden rounded-xl border border-emerald-200 bg-emerald-50/50">
        <div className="flex items-center gap-2 border-b border-emerald-200 bg-emerald-100/60 px-4 py-2">
          <span className="text-sm">📖</span>
          <span className="text-xs font-medium text-emerald-700">
            讲解{args.point_name ? ` · ${args.point_name}` : ""}
          </span>
        </div>
        <div className="px-4 py-3 text-sm text-zinc-800">
          <Markdown content={text} />
        </div>
      </div>
    );
  },
});

/** 所有 Cognia 后端工具的 Generative UI 渲染器，传给 <CopilotKit renderToolCalls>。 */
export const cogniaToolRenderers = [diagnosisRenderer, knowledgeModelRenderer, explainRenderer];
