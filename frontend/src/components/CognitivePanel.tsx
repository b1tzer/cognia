import type { Session, CognitiveState } from '../types'

interface Props {
  session: Session
}

const STATE_COLOR: Record<CognitiveState, string> = {
  understood: '#4a6b5d',
  partial: '#c08a3e',
  misconceived: '#b0553f',
  insufficient: '#9a938a',
}

const STATE_LABEL: Record<CognitiveState, string> = {
  understood: '理解',
  partial: '半理解',
  misconceived: '误解',
  insufficient: '待诊断',
}

export default function CognitivePanel({ session }: Props) {
  const concepts = session.cognitive?.concepts ?? []

  return (
    <div className="cognitive-panel">
      <div className="panel-header">
        <div className="panel-title">认知状态</div>
        <div className="panel-sub">随对话动态更新</div>
      </div>

      <div className="concept-list">
        {concepts.map((c) => {
          const color = STATE_COLOR[c.state]
          const pct = Math.round(c.mastery * 100)
          return (
            <div key={c.concept_id} className="concept-row">
              <div className="concept-row-top">
                <span className="concept-name">{c.concept_name}</span>
                <span className="concept-state" style={{ color }}>
                  {STATE_LABEL[c.state]}
                </span>
              </div>
              {c.evidence_count > 0 ? (
                <div className="concept-bar">
                  <div
                    className="concept-bar-fill"
                    style={{ width: `${pct}%`, background: color }}
                  />
                </div>
              ) : (
                <div className="concept-bar concept-bar-empty" aria-label="待诊断" />
              )}
              <div className="concept-meta">
                {c.evidence_count > 0 ? `已反馈 ${c.evidence_count} 次` : '尚未反馈'}
                {c.consecutive_failures >= 3 ? ' · 连续卡壳' : ''}
              </div>
            </div>
          )
        })}
      </div>

      {session.cognitive?.concepts.some((c) => c.state === 'misconceived') && (
        <div className="panel-note warn">检测到误解，导师会优先引导纠错。</div>
      )}
      {concepts.some((c) => c.consecutive_failures >= 3) && (
        <div className="panel-note warn">有概念连续卡壳，导师可能回溯到前置概念巩固。</div>
      )}    </div>
  )
}
