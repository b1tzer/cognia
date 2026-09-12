import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getAtlas, getAtlasNeighbors } from '../api'
import type {
  AtlasView as AtlasData,
  AtlasNeighborsResponse,
  RelationType,
  CognitiveState,
} from '../types'
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

const RELATION_STYLE: Record<RelationType, { stroke: string; dash: string; arrow: boolean }> = {
  'is-a': { stroke: '#4a6b5d', dash: '', arrow: false },
  related: { stroke: '#9a938a', dash: '6 4', arrow: false },
  prerequisite: { stroke: '#c08a3e', dash: '', arrow: true },
}

const RELATION_LABEL: Record<RelationType, string> = {
  'is-a': 'is-a',
  related: '相关',
  prerequisite: '前置',
}

const MAX_DEPTH = 6

interface ViewTransform {
  x: number
  y: number
  k: number
}

export default function AtlasView({ onBack }: Props) {
  const [data, setData] = useState<AtlasData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // 焦点模式（#37：点击节点展开周边关联）
  const [focusId, setFocusId] = useState<string | null>(null)
  const [depth, setDepth] = useState(1)
  const [neighbors, setNeighbors] = useState<AtlasNeighborsResponse | null>(null)
  const [neighborsLoading, setNeighborsLoading] = useState(false)
  const [neighborsError, setNeighborsError] = useState<string | null>(null)

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

  // 焦点概念切换时按当前深度拉取周边关联
  useEffect(() => {
    if (!focusId) {
      setNeighbors(null)
      return
    }
    let cancelled = false
    setNeighborsLoading(true)
    setNeighborsError(null)
    getAtlasNeighbors(focusId, depth)
      .then((r) => {
        if (!cancelled) setNeighbors(r)
      })
      .catch((e: unknown) => {
        if (!cancelled) setNeighborsError(e instanceof Error ? e.message : '加载周边关联失败')
      })
      .finally(() => {
        if (!cancelled) setNeighborsLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [focusId, depth])

  const concepts = data?.concepts ?? []
  const relations = data?.relations ?? []

  const conceptById = useMemo(() => {
    const m = new Map<string, (typeof concepts)[number]>()
    for (const c of concepts) m.set(c.id, c)
    return m
  }, [concepts])

  const nodeMeta = useCallback(
    (id: string): { name: string; summary: string; mastery: number; state: CognitiveState } => {
      const c = conceptById.get(id)
      if (c) return { name: c.name, summary: c.summary, mastery: c.mastery, state: c.state }
      return { name: id, summary: '', mastery: 0, state: 'insufficient' }
    },
    [conceptById],
  )

  const globalNodes = useMemo(() => concepts.map((c) => ({ id: c.id, mastery: c.mastery })), [concepts])
  const globalEdges = useMemo(
    () => relations.map((r) => ({ from: r.from, to: r.to, relation_type: r.relation_type, depth: 0 })),
    [relations],
  )

  const neighborList = neighbors?.neighbors ?? []
  const subNodeIds = useMemo(() => {
    if (!focusId) return []
    const ids = new Set<string>([focusId])
    for (const n of neighborList) {
      ids.add(n.node)
      ids.add(n.from)
      ids.add(n.to)
    }
    return [...ids]
  }, [focusId, neighborList])

  const subNodes = useMemo(
    () => subNodeIds.map((id) => ({ id, mastery: conceptById.get(id)?.mastery ?? 0 })),
    [subNodeIds, conceptById],
  )
  const subEdges = useMemo(
    () => neighborList.map((n) => ({ from: n.from, to: n.to, relation_type: n.relation_type, depth: n.depth })),
    [neighborList],
  )

  const renderNodes = focusId ? subNodes : globalNodes
  const renderEdges = focusId ? subEdges : globalEdges
  const layout = useMemo(() => forceLayout(renderNodes, renderEdges), [renderNodes, renderEdges])

  // 数据 / 尺寸 / 模式变化时自适应居中
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

  const onNodeClick = useCallback(
    (id: string) => {
      if (nodeDragRef.current?.moved) return
      setFocusId(id)
      setDepth(1)
      setDrag({})
    },
    [],
  )

  const clearFocus = useCallback(() => {
    setFocusId(null)
    setDrag({})
  }, [])

  const hasNodes = concepts.length > 0
  const focusConcept = focusId ? conceptById.get(focusId) : undefined
  const isIsolated = !!focusId && !neighborsLoading && !neighborsError && neighborList.length === 0

  return (
    <div className="atlas-view">
      <div className="atlas-toolbar">
        <button className="btn btn-ghost" onClick={onBack}>← 返回</button>
        {focusId && (
          <button className="btn btn-ghost" onClick={clearFocus}>← 返回全局版图</button>
        )}
        <div className="atlas-toolbar-title">
          <span className="atlas-title">{focusId ? '概念周边' : '认知版图'}</span>
          <span className="atlas-meta">
            {loading
              ? '加载中…'
              : focusId
                ? `「${focusConcept?.name ?? focusId}」 · ${neighborList.length} 条关联`
                : `${concepts.length} 概念 · ${relations.length} 关系`}
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
            <defs>
              <marker
                id="arrow-prereq"
                viewBox="0 0 10 10"
                refX="10"
                refY="5"
                markerWidth="7"
                markerHeight="7"
                orient="auto-start-reverse"
              >
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#c08a3e" />
              </marker>
            </defs>
            <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
              {renderEdges.map((e, i) => {
                const a = posOf(e.from)
                const b = posOf(e.to)
                if (!a || !b) return null
                const st = RELATION_STYLE[e.relation_type]
                return (
                  <line
                    key={i}
                    x1={a.x}
                    y1={a.y}
                    x2={b.x}
                    y2={b.y}
                    stroke={st.stroke}
                    strokeDasharray={st.dash || undefined}
                    strokeWidth={1.4}
                    className="atlas-edge"
                    markerEnd={st.arrow ? 'url(#arrow-prereq)' : undefined}
                  />
                )
              })}
              {renderNodes.map((n) => {
                const pos = posOf(n.id)
                if (!pos) return null
                const meta = nodeMeta(n.id)
                const isFocus = n.id === focusId
                const color = STATE_COLOR[meta.state] ?? STATE_COLOR.insufficient
                return (
                  <g
                    key={n.id}
                    className={'atlas-node' + (isFocus ? ' atlas-node-focus' : '')}
                    transform={`translate(${pos.x} ${pos.y})`}
                    onPointerDown={(e) => onNodePointerDown(e, n.id)}
                    onPointerMove={(e) => onNodePointerMove(e, n.id)}
                    onPointerUp={onNodePointerUp}
                    onPointerCancel={onNodePointerUp}
                    onClick={() => onNodeClick(n.id)}
                  >
                    <circle className="atlas-node-ring" r={isFocus ? 20 : 14} stroke={color} />
                    <circle className="atlas-node-fill" r={isFocus ? 11 : 8} fill={color} />
                    <text className="atlas-node-name" y={isFocus ? 36 : 30} textAnchor="middle">
                      {meta.name.length > 12 ? meta.name.slice(0, 12) + '…' : meta.name}
                    </text>
                  </g>
                )
              })}
            </g>
          </svg>
        )}

        {!loading && !error && hasNodes && (
          <div className="atlas-legend">
            {(Object.keys(RELATION_LABEL) as RelationType[]).map((t) => {
              const st = RELATION_STYLE[t]
              return (
                <span className="atlas-legend-item" key={t}>
                  <svg width="22" height="10" viewBox="0 0 22 10">
                    <line
                      x1="1"
                      y1="5"
                      x2="21"
                      y2="5"
                      stroke={st.stroke}
                      strokeDasharray={st.dash || undefined}
                      strokeWidth="1.6"
                      markerEnd={st.arrow ? 'url(#arrow-prereq)' : undefined}
                    />
                  </svg>
                  {RELATION_LABEL[t]}
                </span>
              )
            })}
          </div>
        )}

        {focusId && focusConcept && (
          <div className="atlas-focus-panel">
            <div className="atlas-focus-head">
              <span className="atlas-focus-name">{focusConcept.name}</span>
              <button className="atlas-focus-close" onClick={clearFocus} aria-label="关闭">×</button>
            </div>
            {focusConcept.summary && (
              <div className="atlas-focus-summary">{focusConcept.summary}</div>
            )}

            <div className="atlas-depth">
              <label className="atlas-depth-label">邻接深度</label>
              <input
                type="range"
                min={1}
                max={MAX_DEPTH}
                value={depth}
                onChange={(e) => setDepth(Number(e.target.value))}
              />
              <span className="atlas-depth-value">{depth} 层</span>
            </div>

            <div className="atlas-focus-meta">
              {neighborsLoading ? (
                <span>加载周边关联…</span>
              ) : neighborsError ? (
                <span className="atlas-hint-error">{neighborsError}</span>
              ) : isIsolated ? (
                <span className="atlas-isolated">孤立概念 · 暂无关联</span>
              ) : (
                <span>{neighborList.length} 条关联关系</span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
