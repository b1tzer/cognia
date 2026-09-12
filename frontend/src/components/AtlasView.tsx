import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getAtlas } from '../api'
import type { AtlasView as AtlasData } from '../types'
import { forceLayout } from '../atlasLayout'

interface Props {
  onBack: () => void
}

const STATE_COLOR: Record<string, string> = {
  understood: '#4a6b5d',
  partial: '#c08a3e',
  misconceived: '#b0553f',
  insufficient: '#9a938a',
}

interface ViewTransform {
  x: number
  y: number
  k: number
}

export default function AtlasView({ onBack }: Props) {
  const [data, setData] = useState<AtlasData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })

  const [view, setView] = useState<ViewTransform>({ x: 0, y: 0, k: 1 })
  const [drag, setDrag] = useState<Record<string, { x: number; y: number }>>({})

  const panRef = useRef<{ startX: number; startY: number; viewX: number; viewY: number } | null>(null)
  const nodeDragRef = useRef<{ id: string; startX: number; startY: number; origX: number; origY: number; moved: boolean } | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await getAtlas())
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : '加载版图失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  // 容器尺寸测量
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const update = () => setSize({ w: el.clientWidth, h: el.clientHeight })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const concepts = data?.concepts ?? []
  const relations = data?.relations ?? []

  const nodes = useMemo(() => concepts.map((c) => ({ id: c.id, mastery: c.mastery })), [concepts])
  const edges = useMemo(() => relations.map((r) => ({ from: r.from, to: r.to })), [relations])
  const layout = useMemo(() => forceLayout(nodes, edges), [nodes, edges])

  // 数据 / 尺寸变化时自适应居中（拖拽覆盖后的位置仍保留）
  useEffect(() => {
    if (size.w === 0 || size.h === 0) return
    const pad = 60
    const k = Math.min((size.w - pad) / layout.width, (size.h - pad) / layout.height, 1.6)
    setView({
      k,
      x: (size.w - layout.width * k) / 2,
      y: (size.h - layout.height * k) / 2,
    })
  }, [size, layout])

  const posOf = useCallback(
    (id: string) => drag[id] ?? layout.positions.get(id),
    [drag, layout],
  )

  // 缩放：以光标为中心
  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15
    setView((v) => {
      const nk = Math.min(Math.max(v.k * factor, 0.2), 4)
      const gx = (sx - v.x) / v.k
      const gy = (sy - v.y) / v.k
      return { k: nk, x: sx - gx * nk, y: sy - gy * nk }
    })
  }, [])

  // 背景平移
  const onBackgroundPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (e.button !== 0) return
      panRef.current = { startX: e.clientX, startY: e.clientY, viewX: view.x, viewY: view.y }
      ;(e.currentTarget as Element).setPointerCapture?.(e.pointerId)
    },
    [view],
  )

  const onBackgroundPointerMove = useCallback((e: React.PointerEvent) => {
    const pan = panRef.current
    if (!pan) return
    setView((v) => ({ ...v, x: pan.viewX + (e.clientX - pan.startX), y: pan.viewY + (e.clientY - pan.startY) }))
  }, [])

  const onBackgroundPointerUp = useCallback(() => {
    panRef.current = null
  }, [])

  // 节点拖拽
  const onNodePointerDown = useCallback(
    (e: React.PointerEvent, id: string) => {
      e.stopPropagation()
      if (e.button !== 0) return
      const current = drag[id] ?? layout.positions.get(id) ?? { x: 0, y: 0 }
      nodeDragRef.current = { id, startX: e.clientX, startY: e.clientY, origX: current.x, origY: current.y, moved: false }
      ;(e.currentTarget as Element).setPointerCapture?.(e.pointerId)
    },
    [drag, layout],
  )

  const onNodePointerMove = useCallback(
    (e: React.PointerEvent, id: string) => {
      const nd = nodeDragRef.current
      if (!nd || nd.id !== id) return
      const dx = (e.clientX - nd.startX) / view.k
      const dy = (e.clientY - nd.startY) / view.k
      if (Math.abs(dx) + Math.abs(dy) > 0.5) nd.moved = true
      setDrag((prev) => ({ ...prev, [id]: { x: nd.origX + dx, y: nd.origY + dy } }))
    },
    [view.k],
  )

  const onNodePointerUp = useCallback(() => {
    nodeDragRef.current = null
  }, [])

  const hasNodes = concepts.length > 0

  return (
    <div className="atlas-view">
      <div className="atlas-toolbar">
        <button className="btn btn-ghost" onClick={onBack}>← 返回</button>
        <div className="atlas-toolbar-title">
          <span className="atlas-title">认知版图</span>
          <span className="atlas-meta">
            {loading ? '加载中…' : `${concepts.length} 概念 · ${relations.length} 关系`}
          </span>
        </div>
      </div>

      <div
        className="atlas-canvas"
        ref={containerRef}
        onWheel={onWheel}
        onPointerDown={onBackgroundPointerDown}
        onPointerMove={onBackgroundPointerMove}
        onPointerUp={onBackgroundPointerUp}
        onPointerCancel={onBackgroundPointerUp}
      >
        {loading && <div className="atlas-hint">正在加载认知版图…</div>}
        {!loading && error && <div className="atlas-hint atlas-hint-error">{error}</div>}
        {!loading && !error && !hasNodes && (
          <div className="atlas-empty">
            <div className="atlas-empty-title">认知版图还是空的</div>
            <div className="atlas-empty-sub">开始学习后，概念会沉淀到全局版图。</div>
          </div>
        )}
        {!loading && !error && hasNodes && size.w > 0 && (
          <svg className="atlas-svg" style={{ cursor: panRef.current ? 'grabbing' : 'grab' }}>
            <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
              {relations.map((r, i) => {
                const a = posOf(r.from)
                const b = posOf(r.to)
                if (!a || !b) return null
                return (
                  <line
                    key={i}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    className="atlas-edge"
                  />
                )
              })}
              {concepts.map((c) => {
                const pos = posOf(c.id)
                if (!pos) return null
                const color = STATE_COLOR[c.state] ?? STATE_COLOR.insufficient
                return (
                  <g
                    key={c.id}
                    className="atlas-node"
                    transform={`translate(${pos.x} ${pos.y})`}
                    onPointerDown={(e) => onNodePointerDown(e, c.id)}
                    onPointerMove={(e) => onNodePointerMove(e, c.id)}
                    onPointerUp={onNodePointerUp}
                    onPointerCancel={onNodePointerUp}
                  >
                    <circle className="atlas-node-ring" r={14} stroke={color} />
                    <circle className="atlas-node-fill" r={8} fill={color} />
                    <text className="atlas-node-name" y={30} textAnchor="middle">
                      {c.name.length > 12 ? c.name.slice(0, 12) + '…' : c.name}
                    </text>
                  </g>
                )
              })}
            </g>
          </svg>
        )}
      </div>
    </div>
  )
}
