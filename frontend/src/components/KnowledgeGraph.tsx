import { useMemo } from 'react'
import type { KnowledgeModel, CognitiveModel, CognitiveState } from '../types'

interface Props {
  knowledge: KnowledgeModel
  cognitive: CognitiveModel | null
  focusId?: string
}

const STATE_COLOR: Record<CognitiveState, string> = {
  understood: '#4a6b5d',   // 墨绿：掌握/点亮
  partial: '#c08a3e',      // 琥珀：半理解
  misconceived: '#b0553f', // 陶土红：误解
  insufficient: '#9a938a', // 暖灰：信息不足
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
  // 横向 DAG：层数决定宽度（左→右），层内节点数决定高度（纵向堆叠）
  const width = (maxDepth + 1) * NODE_W + maxDepth * GAP_X + PAD * 2
  const height = maxLayerCount * NODE_H + (maxLayerCount - 1) * GAP_Y + PAD * 2

  const positions = new Map<string, { x: number; y: number }>()
  for (const [d, ids] of layers.entries()) {
    const layerHeight = ids.length * NODE_H + (ids.length - 1) * GAP_Y
    const startY = (height - layerHeight) / 2
    ids.forEach((id, i) => {
      positions.set(id, {
        x: PAD + d * (NODE_W + GAP_X),
        y: startY + i * (NODE_H + GAP_Y),
      })
    })
  }

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
              <path d="M0,0 L10,5 L0,10 z" fill="#b8b0a6" />
            </marker>
          </defs>

          {/* 依赖边 */}
          {edges.map((e, i) => {
            const a = layout.positions.get(e.from)!
            const b = layout.positions.get(e.to)!
            // 左→右布局：从右边缘中点连到左边缘中点
            const x1 = a.x + NODE_W
            const y1 = a.y + NODE_H / 2
            const x2 = b.x
            const y2 = b.y + NODE_H / 2
            return (
              <path
                key={i}
                d={`M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`}
                fill="none"
                stroke="#b8b0a6"
                strokeWidth={1.5}
                markerEnd="url(#arrow)"
                opacity={0.55}
              />
            )
          })}

          {/* 概念节点：圆点星图，随理解点亮 */}
          {knowledge.concepts.map((c) => {
            const pos = layout.positions.get(c.id)!
            const cx = pos.x + NODE_W / 2
            const cy = pos.y + NODE_H / 2
            const m = masteryMap.get(c.id)
            const color = m ? STATE_COLOR[m.state] : STATE_COLOR.insufficient
            const mastery = m ? m.mastery : 0
            const isFocus = focusId === c.id
            const r = isFocus ? 22 : 17
            // 内圈填充半径随掌握度增长（未探索 ~ 空心，掌握 ~ 实心点亮）
            const fillR = r * (0.22 + 0.78 * mastery)
            return (
              <g key={c.id} className="gnode">
                {/* 掌握时的光晕 */}
                {m && m.state === 'understood' && (
                  <circle cx={cx} cy={cy} r={r + 9} fill={color} opacity={0.14} className="gnode-glow" />
                )}
                {/* 外环 */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={r}
                  fill="var(--panel, #fbfaf8)"
                  stroke={color}
                  strokeWidth={isFocus ? 2 : 1.2}
                  opacity={0.5}
                />
                {/* 内圈填充：随 mastery 点亮 */}
                <circle cx={cx} cy={cy} r={fillR} fill={color} className="gnode-fill" />
                {/* 概念名标签 */}
                <text x={cx} y={cy + r + 18} textAnchor="middle" className="gnode-name">
                  {c.name.length > 10 ? c.name.slice(0, 10) + '…' : c.name}
                </text>
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}
