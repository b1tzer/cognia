// 力导向布局（Fruchterman–Reingold 简化版）：为全局版图计算稳定的节点坐标。
// 纯 TypeScript 实现，零第三方依赖（与后端「轻量」决策一致）。

export interface LayoutNode {
  id: string
  mastery: number
}

export interface LayoutEdge {
  from: string
  to: string
}

export interface LayoutResult {
  positions: Map<string, { x: number; y: number }>
  width: number
  height: number
}

// 确定性伪随机（同一节点集 → 同一初始布局，避免每次重渲染抖动）
function mulberry32(seed: number) {
  let a = seed >>> 0
  return function () {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function hashString(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

export function forceLayout(nodes: LayoutNode[], edges: LayoutEdge[]): LayoutResult {
  const positions = new Map<string, { x: number; y: number }>()
  const n = nodes.length
  if (n === 0) return { positions, width: 0, height: 0 }

  const seed = hashString(nodes.map((x) => x.id).sort().join('|'))
  const rnd = mulberry32(seed)

  // 初始：随机散布在圆盘内
  const R = Math.sqrt(n) * 140 + 60
  for (const nd of nodes) {
    const angle = rnd() * Math.PI * 2
    const r = Math.sqrt(rnd()) * R
    positions.set(nd.id, { x: Math.cos(angle) * r, y: Math.sin(angle) * r })
  }

  const area = Math.max(n * 30000, 60000)
  const k = Math.sqrt(area / n)

  // 邻接表（只保留两端都在节点集里的边）
  const adj = new Map<string, string[]>()
  for (const nd of nodes) adj.set(nd.id, [])
  const validEdges = edges.filter((e) => adj.has(e.from) && adj.has(e.to))
  for (const e of validEdges) {
    adj.get(e.from)!.push(e.to)
    adj.get(e.to)!.push(e.from)
  }

  const ITER = Math.min(300, Math.max(60, Math.floor(90 + Math.sqrt(n) * 4)))
  let t = 1.0
  const cool = Math.pow(0.01, 1 / ITER)

  for (let iter = 0; iter < ITER; iter++) {
    const disp = new Map<string, { x: number; y: number }>()
    for (const nd of nodes) disp.set(nd.id, { x: 0, y: 0 })

    // 斥力：节点间两两排斥
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const a = nodes[i]
        const b = nodes[j]
        const pa = positions.get(a.id)!
        const pb = positions.get(b.id)!
        let dx = pa.x - pb.x
        let dy = pa.y - pb.y
        let d2 = dx * dx + dy * dy
        if (d2 < 1) {
          dx = rnd() - 0.5
          dy = rnd() - 0.5
          d2 = 1
        }
        const d = Math.sqrt(d2)
        const f = (k * k) / d
        const fx = (dx / d) * f
        const fy = (dy / d) * f
        disp.get(a.id)!.x += fx
        disp.get(a.id)!.y += fy
        disp.get(b.id)!.x -= fx
        disp.get(b.id)!.y -= fy
      }
    }

    // 弹簧引力：有关系的节点相互拉近
    for (const e of validEdges) {
      const pa = positions.get(e.from)!
      const pb = positions.get(e.to)!
      const dx = pb.x - pa.x
      const dy = pb.y - pa.y
      const d = Math.sqrt(dx * dx + dy * dy) || 1
      const f = (d * d) / k
      const fx = (dx / d) * f
      const fy = (dy / d) * f
      disp.get(e.from)!.x += fx
      disp.get(e.from)!.y += fy
      disp.get(e.to)!.x -= fx
      disp.get(e.to)!.y -= fy
    }

    // 全局弱引力：让孤岛 / 孤立节点不至于飘走
    const g = 0.04
    for (const nd of nodes) {
      const p = positions.get(nd.id)!
      disp.get(nd.id)!.x -= p.x * g
      disp.get(nd.id)!.y -= p.y * g
    }

    // 应用位移（温度冷却 + 单步位移上限）
    for (const nd of nodes) {
      const p = positions.get(nd.id)!
      const d = disp.get(nd.id)!
      const len = Math.sqrt(d.x * d.x + d.y * d.y)
      const cap = t * k * 2
      const s = len > cap ? cap / len : 1
      p.x += d.x * s
      p.y += d.y * s
    }
    t *= cool
  }

  // 归一化到非负坐标并居中
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const p of positions.values()) {
    minX = Math.min(minX, p.x)
    maxX = Math.max(maxX, p.x)
    minY = Math.min(minY, p.y)
    maxY = Math.max(maxY, p.y)
  }
  const pad = 90
  const w = Math.max(maxX - minX, 200) + pad * 2
  const h = Math.max(maxY - minY, 200) + pad * 2
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  for (const nd of nodes) {
    const p = positions.get(nd.id)!
    p.x = p.x - cx + w / 2
    p.y = p.y - cy + h / 2
  }
  return { positions, width: w, height: h }
}
