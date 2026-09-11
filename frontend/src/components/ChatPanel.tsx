import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import type { Message, Session, CognitiveState, TutorAction, LLMTrace } from '../types'

interface Props {
  session: Session
  onSend: (content: string) => void
  onDeleteMessage: (index: number) => void
  onUpdateMessage: (index: number, content: string) => void
  busy: boolean
  error: string | null
}

const STATE_LABEL: Record<CognitiveState, string> = {
  understood: '理解',
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

// 学生快捷操作：一键发送，减少打字、提高操作效率
const QUICK_ACTIONS = [
  { label: '我不懂，解释一下', text: '我还不懂，能解释一下吗？' },
  { label: '我懂了，继续', text: '我懂了，继续下一个' },
  { label: '举个例子', text: '能举个具体的例子吗？' },
  { label: '回到上一个概念', text: '回到上一个概念' },
]

export default function ChatPanel({ session, onSend, onDeleteMessage, onUpdateMessage, busy, error }: Props) {
  const [input, setInput] = useState('')
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [editText, setEditText] = useState('')
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

  const startEdit = (index: number, content: string) => {
    setEditingIndex(index)
    setEditText(content)
  }

  const saveEdit = (index: number) => {
    const t = editText.trim()
    if (!t) return
    onUpdateMessage(index, t)
    setEditingIndex(null)
  }

  const cancelEdit = () => setEditingIndex(null)

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-header-goal">{session.goal}</div>
        {session.status === 'completed' && <span className="badge badge-done">已完成</span>}
      </div>

      <div className="chat-scroll">
        {session.messages.map((m, i) => (
          <Bubble
            key={i}
            index={i}
            msg={m}
            editing={editingIndex === i}
            editText={editText}
            onEditStart={startEdit}
            onEditChange={setEditText}
            onEditSave={saveEdit}
            onEditCancel={cancelEdit}
            onDelete={onDeleteMessage}
          />
        ))}
        {busy && session.messages[session.messages.length - 1]?.role !== 'assistant' && (
          <div className="bubble assistant typing">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <div className="error-tip inline">{error}</div>}

      <div className="quick-actions">
        {QUICK_ACTIONS.map((a) => (
          <button
            key={a.label}
            className="quick-chip"
            onClick={() => onSend(a.text)}
            disabled={busy || session.status === 'completed'}
          >
            {a.label}
          </button>
        ))}
      </div>

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

interface BubbleProps {
  index: number
  msg: Message
  editing: boolean
  editText: string
  onEditStart: (index: number, content: string) => void
  onEditChange: (text: string) => void
  onEditSave: (index: number) => void
  onEditCancel: () => void
  onDelete: (index: number) => void
}

function Bubble({ index, msg, editing, editText, onEditStart, onEditChange, onEditSave, onEditCancel, onDelete }: BubbleProps) {
  const isUser = msg.role === 'user'
  const diag = msg.diagnosis
  return (
    <div className={`bubble-row ${isUser ? 'user' : 'assistant'}`}>
      <div className="bubble-avatar">{isUser ? '我' : '◆'}</div>
      <div className="bubble-main">
        {!isUser && msg.action && (
          <span className="action-tag">{ACTION_LABEL[msg.action]}</span>
        )}
        {editing ? (
          <div className="bubble-edit">
            <textarea
              className="bubble-edit-input"
              value={editText}
              onChange={(e) => onEditChange(e.target.value)}
              autoFocus
            />
            <div className="bubble-edit-actions">
              <button className="btn btn-primary btn-sm" onClick={() => onEditSave(index)}>保存</button>
              <button className="btn btn-ghost btn-sm" onClick={onEditCancel}>取消</button>
            </div>
          </div>
        ) : (
          <div className={`bubble ${isUser ? 'user' : 'assistant'}`}>
            <div className="bubble-text">
              {!isUser && !msg.content ? (
                <span className="typing">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </span>
              ) : (
                <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                  {msg.content}
                </ReactMarkdown>
              )}
            </div>
          </div>
        )}
        {diag && (
          <div className={`diag-tag ${diag.state}`}>
            诊断：{STATE_LABEL[diag.state]}
          </div>
        )}
        {!isUser &&
          msg.decision?.reasons?.pedagogical_intent &&
          !msg.decision.reasons.pedagogical_intent.includes('回退') && (
            <div className="decision-tag">🎯 {msg.decision.reasons.pedagogical_intent}</div>
          )}
        {!isUser && msg.trace && msg.trace.length > 0 && (
          <TracePanel trace={msg.trace} />
        )}
      </div>
      {!editing && (
        <div className="bubble-tools">
          {isUser && (
            <button className="bubble-tool" title="编辑" onClick={() => onEditStart(index, msg.content)}>
              ✎
            </button>
          )}
          <button className="bubble-tool" title="删除" onClick={() => {
            if (window.confirm('删除这条消息？')) onDelete(index)
          }}>
            ✕
          </button>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// 「思考过程」折叠面板：展示每层 LLM 调用的 prompt 与原始输出
// 参考 Cursor/Claude 的「可折叠过程记录」模式：默认收起，点开逐层查看
// ---------------------------------------------------------------------------
function TracePanel({ trace }: { trace: LLMTrace[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="trace-panel">
      <button className="trace-toggle" onClick={() => setOpen(!open)}>
        <span className="trace-toggle-icon">🧠</span>
        <span className="trace-toggle-label">思考过程</span>
        <span className="trace-toggle-count">{trace.length} 步</span>
        <span className="trace-chevron">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="trace-list">
          {trace.map((t, i) => (
            <TraceStep key={i} trace={t} index={i} />
          ))}
        </div>
      )}
    </div>
  )
}

function TraceStep({ trace, index }: { trace: LLMTrace; index: number }) {
  return (
    <details className="trace-step">
      <summary>
        <span className="trace-step-index">{index + 1}</span>
        <span className="trace-step-label">{trace.label}</span>
        <span className="trace-step-model">{trace.model}</span>
      </summary>
      <div className="trace-body">
        <div className="trace-section">
          <div className="trace-section-title">System Prompt</div>
          <pre className="trace-pre">{trace.system}</pre>
        </div>
        <div className="trace-section">
          <div className="trace-section-title">User Prompt</div>
          <pre className="trace-pre">{trace.user}</pre>
        </div>
        <div className="trace-section">
          <div className="trace-section-title">AI 输出</div>
          <pre className="trace-pre">{trace.output}</pre>
        </div>
        {trace.usage && (
          <div className="trace-usage">
            tokens：{trace.usage.prompt_tokens} 输入 + {trace.usage.completion_tokens} 输出 = {trace.usage.total_tokens}
          </div>
        )}
      </div>
    </details>
  )
}
