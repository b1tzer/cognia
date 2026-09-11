import type { Session, CognitiveState } from '../types'

interface Props {
  session: Session
}

const STATE_COLOR: Record<CognitiveState, string> = {
  understood: '#22c55e',
  partial: '#f59e0b',
  misconceived: '#ef4444',
  insufficient: '#94a3b8',
}

const STATE_LABEL: Record<CognitiveState, string> = {
  understood: '理解',
  partial: '半理解',
  misconceived: '误解',
  insufficient: '待诊断',
}

export default function CognitivePanel({ session }: Props) {
  const concepts = session.cognitive?.concepts ?? []
  const mastered = concepts.filter((c) => c.mastery >= 0.8).length
  const progress = concepts.length ? Math.round((mastered / concepts.length) * 100) : 0

  return (
    <div className="cognitive-panel">
      <div className="panel-header">
        <div className="panel-title">认知状态</div>
        <div className="panel-sub">{mastered}/{concepts.length} 已掌握</div>
      </div>

      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="progress-label">整体掌握度 {progress}%</div>

      <div className="concept-list">
        {concepts.map((c) => {
          const color = STATE_COLOR[c.state]
          return (
            <div key={c.concept_id} className="concept-row">
              <div className="concept-row-top">
                <span className="concept-name">{c.concept_name}</span>
                <span className="concept-state" style={{ color }}>
                  {STATE_LABEL[c.state]}
                </span>
              </div>
              <div className="concept-bar">
                <div
                  className="concept-bar-fill"
                  style={{ width: `${Math.round(c.mastery * 100)}%`, background: color }}
                />
              </div>
              <div className="concept-meta">
                掌握 {(c.mastery * 100).toFixed(0)}% · 证据 {c.evidence_count}
              </div>
            </div>
          )
        })}
      </div>

      {session.cognitive?.concepts.some((c) => c.state === 'misconceived') && (
        <div className="panel-note warn">
          ⚠️ 检测到误解，导师会优先引导纠错。
        </div>
      )}
    </div>
  )
}
