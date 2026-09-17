"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

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
  { fill: string; stroke: string; dash?: string; label: string; glow: boolean }
> = {
  mastered: { fill: "#18181b", stroke: "#18181b", label: "已掌握", glow: true }, // 实心点亮 + 光晕
  partial: { fill: "#d4d4d8", stroke: "#18181b", label: "部分掌握", glow: false }, // 半填充
  misconception: { fill: "#f59e0b", stroke: "#b45309", label: "错误理解", glow: false }, // 偏色强调
  unknown: { fill: "#71717a", stroke: "#3f3f46", label: "盲区", glow: false }, // 深灰实心
  unassessed: { fill: "#ffffff", stroke: "#a1a1aa", dash: "4 3", label: "未探索", glow: false }, // 虚点
};

function stateOf(proficiencies: Record<string, string>, pointId: string): string {
  return proficiencies[pointId] || "unassessed";
}

// 计算每个知识点的依赖深度（无前置 = 0，即根节点/恒星）
function computeDepth(points: KnowledgeMapPoint[]): Map<string, number> {
  const byId = new Map(points.map((p) => [p.id, p]));
  const memo = new Map<string, number>();
  const depth = (id: string): number => {
    if (memo.has(id)) return memo.get(id)!;
    const p = byId.get(id);
    if (!p || !p.prerequisites || p.prerequisites.length === 0) {
      memo.set(id, 0);
      return 0;
    }
    const d = 1 + Math.max(...p.prerequisites.map((pre) => depth(pre)));
    memo.set(id, d);
    return d;
  };
  points.forEach((p) => depth(p.id));
  return memo;
}

type Position = { x: number; y: number; depth: number; outDegree: number };

const RING_GAP = 200; // 相邻环的半径差
const ROOT_SPREAD = 64; // 多个根节点时在中心小圆内的散布半径

// 径向星系布局：根节点居中，依赖深度决定环半径，同环角度均分
function radialLayout(points: KnowledgeMapPoint[]): {
  positions: Map<string, Position>;
  radius: number;
} {
  const depth = computeDepth(points);
  // 出度（后继数）：该点被多少个点作为前置依赖
  const outDegree = new Map<string, number>();
  points.forEach((p) => outDegree.set(p.id, 0));
  points.forEach((p) =>
    (p.prerequisites || []).forEach((pre) => {
      if (outDegree.has(pre)) outDegree.set(pre, outDegree.get(pre)! + 1);
    }),
  );

  const byDepth = new Map<number, KnowledgeMapPoint[]>();
  points.forEach((p) => {
    const d = depth.get(p.id) ?? 0;
    if (!byDepth.has(d)) byDepth.set(d, []);
    byDepth.get(d)!.push(p);
  });

  const positions = new Map<string, Position>();
  const maxDepth = Math.max(0, ...Array.from(byDepth.keys()));

  byDepth.forEach((group, d) => {
    const n = group.length;
    group.forEach((p, i) => {
      let x = 0;
      let y = 0;
      if (d === 0) {
        // 根节点：单个居中；多个均匀散布在中心小圆
        if (n > 1) {
          const a = (i / n) * Math.PI * 2 - Math.PI / 2;
          x = ROOT_SPREAD * Math.cos(a);
          y = ROOT_SPREAD * Math.sin(a);
        }
      } else {
        const r = d * RING_GAP;
        const a = (i / n) * Math.PI * 2 - Math.PI / 2;
        x = r * Math.cos(a);
        y = r * Math.sin(a);
      }
      positions.set(p.id, { x, y, depth: d, outDegree: outDegree.get(p.id) ?? 0 });
    });
  });

  // 画布半径：最外层环 + 标签缓冲
  const radius = Math.max(1, maxDepth) * RING_GAP + 90;
  return { positions, radius };
}

// 节点半径：根节点（恒星）最大，叶子最小
function nodeRadius(pos: Position): number {
  if (pos.depth === 0) return 20;
  if (pos.outDegree === 0) return 10;
  return 14;
}

type ViewTransform = { x: number; y: number; k: number };

function GoalGalaxy({ goal }: { goal: KnowledgeMapGoal }) {
  const { points, proficiencies } = goal;
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [view, setView] = useState<ViewTransform>({ x: 0, y: 0, k: 1 });
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [focusId, setFocusId] = useState<string | null>(null);

  const viewRef = useRef(view);
  viewRef.current = view;

  const panRef = useRef<{
    startX: number;
    startY: number;
    viewX: number;
    viewY: number;
  } | null>(null);
  const dragMovedRef = useRef(false);

  const layout = useMemo(() => radialLayout(points), [points]);
  const byId = useMemo(() => new Map(points.map((p) => [p.id, p])), [points]);

  // 邻居关系：前置 + 后继
  const neighborsOf = useMemo(() => {
    const pre = new Map<string, string[]>();
    const suc = new Map<string, string[]>();
    points.forEach((p) => {
      pre.set(p.id, p.prerequisites || []);
      suc.set(p.id, []);
    });
    points.forEach((p) =>
      (p.prerequisites || []).forEach((pr) => {
        if (suc.has(pr)) suc.get(pr)!.push(p.id);
      }),
    );
    return (id: string) => [...(pre.get(id) || []), ...(suc.get(id) || [])];
  }, [points]);

  // 容器尺寸测量
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // 尺寸 / 布局变化时自适应居中
  useEffect(() => {
    if (size.w === 0 || size.h === 0) return;
    const pad = 50;
    const rawK = Math.min(
      (size.w - pad) / (layout.radius * 2),
      (size.h - pad) / (layout.radius * 2),
      1.4,
    );
    const k = Math.max(rawK, 0.3);
    setView({ x: size.w / 2, y: size.h / 2, k });
  }, [size, layout]);

  // 滚轮缩放（原生监听 + passive:false，阻止页面滚动）
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const v = viewRef.current;
      const nk = Math.min(Math.max(v.k * factor, 0.2), 4);
      const gx = (sx - v.x) / v.k;
      const gy = (sy - v.y) / v.k;
      setView({ k: nk, x: sx - gx * nk, y: sy - gy * nk });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // 背景拖拽平移
  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const v = viewRef.current;
    panRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      viewX: v.x,
      viewY: v.y,
    };
    dragMovedRef.current = false;
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const pan = panRef.current;
    if (!pan) return;
    const dx = e.clientX - pan.startX;
    const dy = e.clientY - pan.startY;
    if (Math.abs(dx) + Math.abs(dy) > 3) dragMovedRef.current = true;
    setView((v) => ({
      ...v,
      x: pan.viewX + dx,
      y: pan.viewY + dy,
    }));
  }, []);

  const onPointerUp = useCallback(() => {
    panRef.current = null;
  }, []);

  const onBackgroundClick = useCallback(() => {
    if (!dragMovedRef.current) setFocusId(null);
  }, []);

  const highlightSet = useMemo(() => {
    if (!hoverId) return null;
    return new Set([hoverId, ...neighborsOf(hoverId)]);
  }, [hoverId, neighborsOf]);

  const focusPoint = focusId ? byId.get(focusId) : undefined;

  return (
    <section className="rounded-xl border border-line bg-surface p-4">
      <h3 className="mb-2 text-base font-semibold text-foreground">{goal.goal}</h3>
      <div
        ref={containerRef}
        className="relative h-[520px] overflow-hidden rounded-lg border border-line bg-gradient-to-br from-surface-muted to-surface"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onClick={onBackgroundClick}
        style={{ cursor: "grab", touchAction: "none" }}
      >
        <style>{`
          @keyframes km-glow {
            0%, 100% { opacity: 0.22; }
            50% { opacity: 0.65; }
          }
        `}</style>
        <svg className="h-full w-full" role="img" aria-label={`${goal.goal} 径向星系`}>
          <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
            {/* 依赖边：前置 → 当前点（目标 mastered 时点亮为实线墨色） */}
            {points.map((p) =>
              (p.prerequisites || []).map((pre) => {
                const from = layout.positions.get(pre);
                const to = layout.positions.get(p.id);
                if (!from || !to) return null;
                const lit = stateOf(proficiencies, p.id) === "mastered";
                return (
                  <line
                    key={`${pre}-${p.id}`}
                    x1={from.x}
                    y1={from.y}
                    x2={to.x}
                    y2={to.y}
                    stroke={lit ? "#18181b" : "#d4d4d8"}
                    strokeWidth={lit ? 2 : 1.4}
                    strokeDasharray={lit ? undefined : "5 4"}
                  />
                );
              }),
            )}
            {/* 节点 */}
            {points.map((p) => {
              const pos = layout.positions.get(p.id);
              if (!pos) return null;
              const state = stateOf(proficiencies, p.id);
              const style = STATE_STYLE[state] ?? STATE_STYLE.unassessed;
              const r = nodeRadius(pos);
              const dimmed = highlightSet ? !highlightSet.has(p.id) : false;
              const isFocus = focusId === p.id;
              return (
                <g
                  key={p.id}
                  transform={`translate(${pos.x} ${pos.y})`}
                  opacity={dimmed ? 0.2 : 1}
                  style={{ transition: "opacity 0.15s ease", cursor: "pointer" }}
                  onMouseEnter={() => setHoverId(p.id)}
                  onMouseLeave={() =>
                    setHoverId((cur) => (cur === p.id ? null : cur))
                  }
                  onClick={(e) => {
                    e.stopPropagation();
                    setFocusId((cur) => (cur === p.id ? null : p.id));
                  }}
                >
                  {style.glow && (
                    <circle
                      r={r + 9}
                      fill="#18181b"
                      style={{ animation: "km-glow 2.4s ease-in-out infinite" }}
                    />
                  )}
                  <circle
                    r={r}
                    fill={style.fill}
                    stroke={style.stroke}
                    strokeWidth={isFocus ? 3 : 2}
                    strokeDasharray={style.dash}
                  />
                  <text
                    y={r + 16}
                    textAnchor="middle"
                    className="fill-zinc-700"
                    fontSize={12}
                    fontWeight={pos.depth === 0 ? 600 : 400}
                  >
                    {p.name.length > 10 ? p.name.slice(0, 10) + "…" : p.name}
                  </text>
                  <text y={r + 30} textAnchor="middle" className="fill-zinc-400" fontSize={10}>
                    {style.label}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {/* 详情卡 */}
        {focusPoint && (
          <div className="absolute right-3 top-3 w-64 rounded-lg border border-line bg-surface p-3 shadow-lg">
            <div className="flex items-start justify-between gap-2">
              <span className="text-sm font-semibold text-foreground">
                {focusPoint.name}
              </span>
              <button
                className="text-muted transition hover:text-foreground"
                onClick={() => setFocusId(null)}
                aria-label="关闭"
              >
                ×
              </button>
            </div>
            {focusPoint.description && (
              <p className="mt-1 text-xs leading-relaxed text-muted">
                {focusPoint.description}
              </p>
            )}
            <div className="mt-2 text-xs text-muted">
              当前状态：
              <span className="ml-1 font-medium text-foreground">
                {STATE_STYLE[stateOf(proficiencies, focusPoint.id)]?.label ??
                  "未探索"}
              </span>
            </div>
            {focusPoint.prerequisites &&
              focusPoint.prerequisites.length > 0 && (
                <div className="mt-1 text-xs text-muted">
                  前置依赖：
                  {focusPoint.prerequisites
                    .map((id) => byId.get(id)?.name ?? id)
                    .join("、")}
                </div>
              )}
          </div>
        )}

        {/* 图例 */}
        <div className="pointer-events-none absolute bottom-2 left-3 flex flex-wrap gap-3 text-[10px] text-muted">
          {Object.keys(STATE_STYLE).map((k) => (
            <span key={k} className="flex items-center gap-1">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{
                  background: STATE_STYLE[k].fill,
                  border: `1px solid ${STATE_STYLE[k].stroke}`,
                }}
              />
              {STATE_STYLE[k].label}
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function KnowledgeMap({ goals }: { goals: KnowledgeMapGoal[] }) {
  if (!goals || goals.length === 0) return null;
  return (
    <div className="space-y-6">
      {goals.map((goal) => (
        <GoalGalaxy key={goal.goal} goal={goal} />
      ))}
    </div>
  );
}
