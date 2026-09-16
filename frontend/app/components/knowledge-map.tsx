"use client";

import { useMemo } from "react";

// 知识版图数据结构（对齐后端 GET /knowledge-map 返回）
export type KnowledgeMapPoint = {
  id: string;
  name: string;
  description: string;
  prerequisites: string[];
};

export type KnowledgeMapGoal = {
  goal: string;
  points: KnowledgeMapPoint[];
  proficiencies: Record<string, string>;
};

export type KnowledgeMapData = {
  user_id: string;
  goals: KnowledgeMapGoal[];
};

// 认知状态 → 视觉映射（章程 §8 Cognitive Atlas：低饱和 + 墨色 + 单一强调色）
const STATE_STYLE: Record<
  string,
  { fill: string; stroke: string; dash?: string; label: string }
> = {
  mastered: { fill: "#18181b", stroke: "#18181b", label: "已掌握" }, // 实心点亮
  partial: { fill: "#d4d4d8", stroke: "#18181b", label: "部分掌握" }, // 缺口半填充
  misconception: { fill: "#f59e0b", stroke: "#b45309", label: "错误理解" }, // 偏色强调
  unknown: { fill: "#71717a", stroke: "#3f3f46", label: "盲区" }, // 深灰实心
  unassessed: { fill: "#ffffff", stroke: "#a1a1aa", dash: "4 3", label: "未探索" }, // 虚点
};

function stateOf(proficiencies: Record<string, string>, pointId: string): string {
  return proficiencies[pointId] || "unassessed";
}

// 计算每个知识点的拓扑层级（前置依赖深度），用于 DAG 分层布局
function computeLevels(points: KnowledgeMapPoint[]): Map<string, number> {
  const byId = new Map(points.map((p) => [p.id, p]));
  const memo = new Map<string, number>();
  const level = (id: string): number => {
    if (memo.has(id)) return memo.get(id)!;
    const p = byId.get(id);
    if (!p || !p.prerequisites || p.prerequisites.length === 0) {
      memo.set(id, 0);
      return 0;
    }
    const l = 1 + Math.max(...p.prerequisites.map((pre) => level(pre)));
    memo.set(id, l);
    return l;
  };
  points.forEach((p) => level(p.id));
  return memo;
}

const NODE_R = 12;
const LEVEL_GAP_X = 180;
const NODE_GAP_Y = 72;
const PAD = 48;

function GoalGraph({ goal }: { goal: KnowledgeMapGoal }) {
  const { points, proficiencies } = goal;

  const layout = useMemo(() => {
    const levels = computeLevels(points);
    // 按层级分组
    const byLevel = new Map<number, KnowledgeMapPoint[]>();
    points.forEach((p) => {
      const lv = levels.get(p.id) ?? 0;
      if (!byLevel.has(lv)) byLevel.set(lv, []);
      byLevel.get(lv)!.push(p);
    });
    // 计算坐标
    const positions = new Map<string, { x: number; y: number }>();
    byLevel.forEach((group, lv) => {
      group.forEach((p, idx) => {
        positions.set(p.id, {
          x: PAD + lv * LEVEL_GAP_X,
          y: PAD + idx * NODE_GAP_Y,
        });
      });
    });
    const maxLevel = Math.max(0, ...Array.from(byLevel.keys()));
    const maxCount = Math.max(1, ...Array.from(byLevel.values()).map((g) => g.length));
    const width = PAD * 2 + maxLevel * LEVEL_GAP_X + NODE_R * 2;
    const height = PAD * 2 + (maxCount - 1) * NODE_GAP_Y + NODE_R * 2;
    return { positions, byLevel, width, height };
  }, [points]);

  return (
    <section className="rounded-xl border border-zinc-200 bg-white p-4">
      <h3 className="mb-2 text-base font-semibold text-zinc-900">{goal.goal}</h3>
      <svg
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        className="w-full"
        role="img"
        aria-label={`${goal.goal} 知识版图`}
      >
        {/* 依赖边：前置 → 当前点 */}
        {points.map((p) =>
          (p.prerequisites || []).map((pre) => {
            const from = layout.positions.get(pre);
            const to = layout.positions.get(p.id);
            if (!from || !to) return null;
            return (
              <line
                key={`${pre}-${p.id}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke="#d4d4d8"
                strokeWidth={1.5}
              />
            );
          }),
        )}
        {/* 节点 + 标签 */}
        {points.map((p) => {
          const pos = layout.positions.get(p.id);
          if (!pos) return null;
          const state = stateOf(proficiencies, p.id);
          const style = STATE_STYLE[state] ?? STATE_STYLE.unassessed;
          return (
            <g key={p.id}>
              <circle
                cx={pos.x}
                cy={pos.y}
                r={NODE_R}
                fill={style.fill}
                stroke={style.stroke}
                strokeWidth={2}
                strokeDasharray={style.dash}
              />
              <text
                x={pos.x}
                y={pos.y + NODE_R + 16}
                textAnchor="middle"
                className="fill-zinc-700"
                fontSize={12}
              >
                {p.name}
              </text>
              <text
                x={pos.x}
                y={pos.y + NODE_R + 30}
                textAnchor="middle"
                className="fill-zinc-400"
                fontSize={10}
              >
                {style.label}
              </text>
            </g>
          );
        })}
      </svg>
    </section>
  );
}

export default function KnowledgeMap({ goals }: { goals: KnowledgeMapGoal[] }) {
  if (!goals || goals.length === 0) return null;
  return (
    <div className="space-y-6">
      {goals.map((goal) => (
        <GoalGraph key={goal.goal} goal={goal} />
      ))}
    </div>
  );
}
