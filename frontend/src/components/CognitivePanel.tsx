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
          return (
            <div key={c.concept_id} className="concept-row">
              <div className="concept-row-top">
                <span className="concept-name">{c.concept_name}</span>
                <span className="concept-state" style={{ color }}>
                  {STATE_LABEL[c.state]}
                </span>
              </div>
            </div>
          )
        })}
      </div>

      {session.cognitive?.concepts.some((c) => c.state === 'misconceived') && (
        <div className="panel-note warn">
          检测到误解，导师会优先引导纠错。
        </div>
      )}
    </div>
  )
}
