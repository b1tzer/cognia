import { useEffect, useRef, useState } from 'react'
import type { Message, Session, CognitiveState, TutorAction } from '../types'

interface Props {
  session: Session
  onSend: (content: string) => void
  busy: boolean
  error: string | null
}

const STATE_LABEL: Record<CognitiveState, string> = {
  understood: '理解 ✓',
  partial: '半理解',
  misconceived: '有误解',
  insufficient: '信息不足',
}

const ACTION_LABEL: Record<TutorAction, string> = {
  probe: '追问',
  explain: '解释',
  correct: '纠错',
  backtrack: '回溯',
  advance: '推进',
}

export default function ChatPanel({ session, onSend, busy, error }: Props) {
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [session.messages.length, busy])

  const submit = () => {
    const t = input.trim()
    if (!t || busy) return
    onSend(t)
    setInput('')
  }

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-header-goal">{session.goal}</div>
        {session.status === 'completed' && <span className="badge badge-done">已完成</span>}
      </div>

      <div className="chat-scroll">
        {session.messages.map((m, i) => (
          <Bubble key={i} msg={m} />
        ))}
        {busy && (
          <div className="bubble assistant typing">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <div className="error-tip inline">{error}</div>}

      <div className="chat-input-bar">
        <textarea
          className="chat-input"
          placeholder={
            session.status === 'completed'
              ? '本目标已完成，点击右上角「新目标」继续'
              : '用自己的话表达你的理解…（Enter 发送，Shift+Enter 换行）'
          }
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
          rows={2}
          disabled={busy || session.status === 'completed'}
        />
        <button
          className="btn btn-primary send-btn"
          onClick={submit}
          disabled={busy || !input.trim() || session.status === 'completed'}
        >
          发送
        </button>
      </div>
    </div>
  )
}

function Bubble({ msg }: { msg: Message }) {
  const isUser = msg.role === 'user'
  const diag = msg.diagnosis
  return (
    <div className={`bubble-row ${isUser ? 'user' : 'assistant'}`}>
      <div className="bubble-avatar">{isUser ? '我' : '◆'}</div>
      <div className="bubble-main">
        {!isUser && msg.action && (
          <span className="action-tag">{ACTION_LABEL[msg.action]}</span>
        )}
        <div className={`bubble ${isUser ? 'user' : 'assistant'}`}>
          <div className="bubble-text">{msg.content}</div>
        </div>
        {diag && (
          <div className={`diag-tag ${diag.state}`}>
            诊断：{STATE_LABEL[diag.state]}
            {diag.confidence > 0 && <span className="diag-conf"> · 置信 {(diag.confidence * 100).toFixed(0)}%</span>}
          </div>
        )}
      </div>
    </div>
  )
}
