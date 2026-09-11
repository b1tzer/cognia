import { useMemo } from 'react'
import type { KnowledgeModel, CognitiveModel, CognitiveState } from '../types'

interface Props {
  knowledge: KnowledgeModel
  cognitive: CognitiveModel | null
  focusId?: string
}

const STATE_COLOR: Record<CognitiveState, string> = {
  understood: '#22c55e',
  partial: '#f59e0b',
  misconceived: '#ef4444',
  insufficient: '#64748b',
}

const NODE_W = 196
const NODE_H = 74
const GAP_X = 44
const GAP_Y = 96
const PAD = 40

interface Layout {
  positions: Map<string, { x: number; y: number }>
  width: number
  height: number
  layers: Map<number, string[]>
}

function computeLayout(knowledge: KnowledgeModel): Layout {
  const byId = new Map(knowledge.concepts.map((c) => [c.id, c]))
  const depth = new Map<string, number>()

  const getDepth = (id: string, visiting: Set<string>): number => {
    if (depth.has(id)) return depth.get(id)!
    if (visiting.has(id)) return 0
    visiting.add(id)
    const c = byId.get(id)
    let d = 0
    for (const p of c?.prerequisites ?? []) {
      if (byId.has(p)) d = Math.max(d, getDepth(p, visiting) + 1)
    }
    visiting.delete(id)
    depth.set(id, d)
    return d
  }
  knowledge.concepts.forEach((c) => getDepth(c.id, new Set()))

  const layers = new Map<number, string[]>()
  let maxDepth = 0
  for (const c of knowledge.concepts) {
    const d = depth.get(c.id) ?? 0
    maxDepth = Math.max(maxDepth, d)
    if (!layers.has(d)) layers.set(d, [])
    layers.get(d)!.push(c.id)
  }

  let maxLayerCount = 0
  for (const ids of layers.values()) maxLayerCount = Math.max(maxLayerCount, ids.length)
  const width = maxLayerCount * NODE_W + (maxLayerCount - 1) * GAP_X + PAD * 2

  const positions = new Map<string, { x: number; y: number }>()
  for (const [d, ids] of layers.entries()) {
    const layerWidth = ids.length * NODE_W + (ids.length - 1) * GAP_X
    const startX = (width - layerWidth) / 2
    ids.forEach((id, i) => {
      positions.set(id, {
        x: startX + i * (NODE_W + GAP_X),
        y: PAD + d * (NODE_H + GAP_Y),
      })
    })
  }

  const height = PAD + maxDepth * (NODE_H + GAP_Y) + NODE_H + PAD
  return { positions, width, height, layers }
}

export default function KnowledgeGraph({ knowledge, cognitive, focusId }: Props) {
  const layout = useMemo(() => computeLayout(knowledge), [knowledge])
  const masteryMap = useMemo(() => {
    const m = new Map<string, { mastery: number; state: CognitiveState }>()
    for (const c of cognitive?.concepts ?? []) {
      m.set(c.concept_id, { mastery: c.mastery, state: c.state })
    }
    return m
  }, [cognitive])

  const byId = useMemo(() => new Map(knowledge.concepts.map((c) => [c.id, c])), [knowledge])

  const edges: Array<{ from: string; to: string }> = []
  for (const c of knowledge.concepts) {
    for (const p of c.prerequisites) {
      if (byId.has(p)) edges.push({ from: p, to: c.id })
    }
  }

  return (
    <div className="graph-wrap">
      <div className="panel-header">
        <div className="panel-title">知识模型</div>
        <div className="panel-sub">{knowledge.concepts.length} 个概念 · {edges.length} 条依赖</div>
      </div>
      <div className="graph-scroll">
        <svg
          width={layout.width}
          height={layout.height}
          viewBox={`0 0 ${layout.width} ${layout.height}`}
        >
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8" />
            </marker>
          </defs>

          {/* 依赖边 */}
          {edges.map((e, i) => {
            const a = layout.positions.get(e.from)!
            const b = layout.positions.get(e.to)!
            const x1 = a.x + NODE_W / 2
            const y1 = a.y + NODE_H
            const x2 = b.x + NODE_W / 2
            const y2 = b.y
            return (
              <path
                key={i}
                d={`M ${x1} ${y1} C ${x1} ${(y1 + y2) / 2}, ${x2} ${(y1 + y2) / 2}, ${x2} ${y2}`}
                fill="none"
                stroke="#94a3b8"
                strokeWidth={1.5}
                markerEnd="url(#arrow)"
                opacity={0.55}
              />
            )
          })}

          {/* 概念节点 */}
          {knowledge.concepts.map((c) => {
            const pos = layout.positions.get(c.id)!
            const m = masteryMap.get(c.id)
            const color = m ? STATE_COLOR[m.state] : '#64748b'
            const pct = m ? Math.round(m.mastery * 100) : 0
            const isFocus = focusId === c.id
            return (
              <g key={c.id} transform={`translate(${pos.x}, ${pos.y})`}>
                <rect
                  x={0}
                  y={0}
                  width={NODE_W}
                  height={NODE_H}
                  rx={12}
                  fill="#1e293b"
                  stroke={isFocus ? '#38bdf8' : color}
                  strokeWidth={isFocus ? 2.5 : 1.5}
                />
                <text x={14} y={26} className="gnode-name" fill="#f1f5f9">
                  {c.name.length > 12 ? c.name.slice(0, 12) + '…' : c.name}
                </text>
                <text x={14} y={44} className="gnode-summary" fill="#94a3b8">
                  {(c.summary || '').length > 16 ? c.summary.slice(0, 16) + '…' : c.summary}
                </text>
                <rect x={14} y={54} width={NODE_W - 28} height={6} rx={3} fill="#334155" />
                <rect x={14} y={54} width={(NODE_W - 28) * (pct / 100)} height={6} rx={3} fill={color} />
                <text x={NODE_W - 14} y={50} textAnchor="end" className="gnode-pct" fill={color}>
                  {pct}%
                </text>
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}
