import { useCallback, useEffect, useState } from 'react'
import * as api from './api'
import type { Session, ChatResponse } from './types'
import GoalInput from './components/GoalInput'
import KnowledgeGraph from './components/KnowledgeGraph'
import ChatPanel from './components/ChatPanel'
import CognitivePanel from './components/CognitivePanel'
import SessionList from './components/SessionList'

export default function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [sessions, setSessions] = useState<Array<any>>([])
  const [aiEnabled, setAiEnabled] = useState<boolean | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [focusId, setFocusId] = useState<string | null>(null)

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
    // 乐观追加用户消息
    setSession((prev) =>
      prev
        ? { ...prev, messages: [...prev.messages, { role: 'user', content }] }
        : prev,
    )
    try {
      const r: ChatResponse = await api.sendChat(session.id, content)
      setFocusId(r.focus_concept?.id ?? null)
      setSession((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          status: r.status,
          cognitive: r.cognitive,
          messages: [
            ...prev.messages,
            { role: 'assistant', content: r.reply, action: r.action, diagnosis: r.diagnosis, decision: r.decision },
          ],
        }
      })
      refreshSessions()
    } catch (e: any) {
      setError(e.message || '发送失败')
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

  if (!session) {
    return (
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <span className="logo">◆</span>
            <span className="brand-name">Cognia</span>
            <span className="brand-tag">AI Learning Agent</span>
          </div>
          <AiBadge aiEnabled={aiEnabled} />
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
            onDeleteMessage={handleDeleteMessage}
            onUpdateMessage={handleUpdateMessage}
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
