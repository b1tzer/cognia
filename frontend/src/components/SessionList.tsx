import { useState } from 'react'
import ConfirmDialog from './ConfirmDialog'

interface SessionMeta {
  id: string
  goal: string
  created_at: string
  updated_at: string
  status: string
}

interface Props {
  sessions: SessionMeta[]
  onSelect: (id: string) => void
  onDelete: (id: string) => void
  activeId: string | null
}

function fmtTime(iso: string): string {
  try {
    const d = new Date(iso)
    const now = new Date()
    const diff = now.getTime() - d.getTime()
    if (diff < 60_000) return '刚刚'
    if (diff < 3600_000) return `${Math.floor(diff / 60_000)} 分钟前`
    if (diff < 86400_000) return `${Math.floor(diff / 3600_000)} 小时前`
    return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
  } catch {
    return ''
  }
}

export default function SessionList({ sessions, onSelect, onDelete, activeId }: Props) {
  const [pendingDelete, setPendingDelete] = useState<{ id: string; goal: string } | null>(null)

  if (sessions.length === 0) {
    return <div className="session-empty">暂无学习记录</div>
  }
  return (
    <div className="session-list">
      <div className="session-title">学习记录</div>
      {sessions.map((s) => (
        <div
          key={s.id}
          className={`session-item ${s.id === activeId ? 'active' : ''}`}
          onClick={() => onSelect(s.id)}
          title={s.goal}
        >
          <div className="session-item-goal">{s.goal}</div>
          <div className="session-item-meta">
            {s.status === 'completed' ? '✅ 已完成' : '⏳ 进行中'} · {fmtTime(s.updated_at)}
          </div>
          <button
            className="session-item-del"
            aria-label={`删除学习目标「${s.goal}」`}
            onClick={(e) => {
              e.stopPropagation()
              setPendingDelete({ id: s.id, goal: s.goal })
            }}
            title="删除学习目标"
          >
            ✕
          </button>
        </div>
      ))}
      {pendingDelete && (
        <ConfirmDialog
          title="删除学习目标"
          message={`确定删除「${pendingDelete.goal}」？此操作不可撤销。`}
          confirmLabel="删除"
          danger
          onConfirm={() => {
            onDelete(pendingDelete.id)
            setPendingDelete(null)
          }}
          onCancel={() => setPendingDelete(null)}
        />
      )}
    </div>
  )
}
