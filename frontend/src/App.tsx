import { useCallback, useEffect, useState } from 'react'
import * as api from './api'
import type { Session, ChatResponse } from './types'
import GoalInput from './components/GoalInput'
import KnowledgeGraph from './components/KnowledgeGraph'
import ChatPanel from './components/ChatPanel'
import CognitivePanel from './components/CognitivePanel'
import SessionList from './components/SessionList'
import AtlasView from './components/AtlasView'

export default function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [sessions, setSessions] = useState<Array<any>>([])
  const [aiEnabled, setAiEnabled] = useState<boolean | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [focusId, setFocusId] = useState<string | null>(null)
  const [showAtlas, setShowAtlas] = useState(false)

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await api.listSessions())
    } catch {
      /* ignore */
    }
  }, [])

  useEffect(() => {
    api.getHealth().then((h) => setAiEnabled(h.ai_enabled)).catch(() => {})
    refreshSessions()
  }, [refreshSessions])

  const handleStart = useCallback(async (goal: string) => {
    setBusy(true)
    setError(null)
    try {
      const s = await api.createSession(goal)
      setSession(s)
      setFocusId(s.focus_concept?.id ?? null)
      refreshSessions()
    } catch (e: any) {
      setError(e.message || '创建会话失败')
    } finally {
      setBusy(false)
    }
  }, [refreshSessions])

  const handleSelect = useCallback(async (id: string) => {
    setBusy(true)
    setError(null)
    try {
      const s = await api.getSession(id)
      setSession(s)
      setFocusId(null)
    } catch (e: any) {
      setError(e.message || '加载会话失败')
    } finally {
      setBusy(false)
    }
  }, [])

  const handleSend = useCallback(async (content: string) => {
    if (!session) return
    setBusy(true)
    setError(null)
    // 乐观追加用户消息 + 一个空内容的 assistant 占位消息（流式打字机填充）
    setSession((prev) =>
      prev
        ? {
            ...prev,
            messages: [
              ...prev.messages,
              { role: 'user', content },
              { role: 'assistant', content: '', action: null, diagnosis: null, decision: null, trace: [] },
            ],
          }
        : prev,
    )
    try {
      await api.sendChatStream(session.id, content, {
        onTrace: (step, index) => {
          setSession((prev) => {
            if (!prev) return prev
            const msgs = [...prev.messages]
            const last = msgs[msgs.length - 1]
            if (last && last.role === 'assistant') {
              const trace = [...(last.trace || [])]
              trace[index] = step
              msgs[msgs.length - 1] = { ...last, trace }
            }
            return { ...prev, messages: msgs }
          })
        },
        onToken: (token) => {
          setSession((prev) => {
            if (!prev) return prev
            const msgs = [...prev.messages]
            const last = msgs[msgs.length - 1]
            if (last && last.role === 'assistant') {
              msgs[msgs.length - 1] = { ...last, content: last.content + token }
            }
            return { ...prev, messages: msgs }
          })
        },
        onDone: (r: ChatResponse) => {
          // 改目标/澄清这类结构性变更：直接重载会话，拿到完整的新 messages/knowledge/cognitive
          if (r.stage) {
            api.getSession(session.id).then((s) => {
              setSession(s)
              setFocusId(s.focus_concept?.id ?? null)
            }).catch(() => {})
            refreshSessions()
            return
          }
          setFocusId(r.focus_concept?.id ?? null)
          setSession((prev) => {
            if (!prev) return prev
            const msgs = [...prev.messages]
            const last = msgs[msgs.length - 1]
            if (last && last.role === 'assistant') {
              msgs[msgs.length - 1] = {
                ...last,
                content: r.reply,
                action: r.action,
                diagnosis: r.diagnosis,
                decision: r.decision,
                trace: r.trace,
              }
            }
            return {
              ...prev,
              status: r.status,
              cognitive: r.cognitive,
              messages: msgs,
            }
          })
          refreshSessions()
        },
        onError: (e) => {
          setError(e.message || '发送失败')
        },
      })
    } catch (e: any) {
      setError(e.message || '发送失败')
    } finally {
      setBusy(false)
    }
  }, [session, refreshSessions])

  const handleConfirmGoal = useCallback(async (goal: string) => {
    if (!session) return
    setBusy(true)
    setError(null)
    try {
      const s = await api.confirmGoal(session.id, goal)
      setSession(s)
      setFocusId(s.focus_concept?.id ?? null)
      refreshSessions()
    } catch (e: any) {
      setError(e.message || '确认目标失败')
    } finally {
      setBusy(false)
    }
  }, [session, refreshSessions])

  const handleUpdateGoal = useCallback(async (goal: string) => {
    if (!session) return
    setBusy(true)
    setError(null)
    try {
      const s = await api.updateGoal(session.id, goal)
      setSession(s)
      setFocusId(s.focus_concept?.id ?? null)
      refreshSessions()
    } catch (e: any) {
      setError(e.message || '修改目标失败')
    } finally {
      setBusy(false)
    }
  }, [session, refreshSessions])

  const handleNew = useCallback(() => {
    setSession(null)
    setFocusId(null)
    setError(null)
  }, [])

  const handleDeleteSession = useCallback(async (id: string) => {
    try {
      await api.deleteSession(id)
      if (session?.id === id) setSession(null)
      refreshSessions()
    } catch (e: any) {
      setError(e.message || '删除失败')
    }
  }, [session, refreshSessions])

  const handleDeleteMessage = useCallback(async (index: number) => {
    if (!session) return
    try {
      const r = await api.deleteMessage(session.id, index)
      setSession((prev) => (prev ? { ...prev, messages: r.messages } : prev))
    } catch (e: any) {
      setError(e.message || '删除消息失败')
    }
  }, [session])

  const handleUpdateMessage = useCallback(async (index: number, content: string) => {
    if (!session) return
    try {
      const r = await api.updateMessage(session.id, index, content)
      setSession((prev) => (prev ? { ...prev, messages: r.messages } : prev))
    } catch (e: any) {
      setError(e.message || '修改消息失败')
    }
  }, [session])

  const handleRegenerate = useCallback(async (index: number) => {
    if (!session) return
    setBusy(true)
    setError(null)
    try {
      const s = await api.regenerateMessage(session.id, index)
      setSession(s)
      setFocusId(s.focus_concept?.id ?? null)
    } catch (e: any) {
      setError(e.message || '重新生成失败')
    } finally {
      setBusy(false)
    }
  }, [session])

  if (showAtlas) {
    return <AtlasView onBack={() => setShowAtlas(false)} />
  }

  if (!session) {
    return (
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <span className="logo">◆</span>
            <span className="brand-name">Cognia</span>
            <span className="brand-tag">AI Learning Agent</span>
          </div>
          <div className="topbar-actions">
            <button className="btn btn-ghost" onClick={() => setShowAtlas(true)}>认知版图</button>
            <AiBadge aiEnabled={aiEnabled} />
          </div>
        </header>
        <GoalInput onStart={handleStart} busy={busy} error={error} />
        {sessions.length > 0 && (
          <div className="recent">
            <h3>继续学习</h3>
            <SessionList sessions={sessions} onSelect={handleSelect} onDelete={handleDeleteSession} activeId={null} />
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="app app-main">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◆</span>
          <span className="brand-name">Cognia</span>
          <span className="brand-tag">AI Learning Agent</span>
        </div>
        <div className="topbar-actions">
          <AiBadge aiEnabled={aiEnabled} />
          <button className="btn btn-ghost" onClick={() => setShowAtlas(true)}>认知版图</button>
          <button className="btn btn-ghost" onClick={handleNew}>新目标</button>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <SessionList sessions={sessions} onSelect={handleSelect} onDelete={handleDeleteSession} activeId={session.id} />
        </aside>

        <main className="chat-col">
          <ChatPanel
            session={session}
            onSend={handleSend}
            onConfirmGoal={handleConfirmGoal}
            onUpdateGoal={handleUpdateGoal}
            onDeleteMessage={handleDeleteMessage}
            onUpdateMessage={handleUpdateMessage}
            onRegenerate={handleRegenerate}
            busy={busy}
            error={error}
          />
        </main>

        <aside className="insight-col">
          <CognitivePanel session={session} />
        </aside>
      </div>

      {session.knowledge && (
        <div className="graph-dock">
          <KnowledgeGraph
            knowledge={session.knowledge}
            cognitive={session.cognitive}
            focusId={focusId ?? undefined}
          />
        </div>
      )}
    </div>
  )
}

function AiBadge({ aiEnabled }: { aiEnabled: boolean | null }) {
  if (aiEnabled === null) return null
  return (
    <span className={`badge ${aiEnabled ? 'badge-on' : 'badge-off'}`}>
      {aiEnabled ? 'AI 引擎 · 已连接' : '离线诊断模式'}
    </span>
  )
}
