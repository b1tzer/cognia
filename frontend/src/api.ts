import type { ChatResponse, Session } from './types'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `请求失败 (${res.status})`)
  }
  return res.json() as Promise<T>
}

export function createSession(goal: string): Promise<Session> {
  return request<Session>('/api/sessions', {
    method: 'POST',
    body: JSON.stringify({ goal }),
  })
}

export function listSessions(): Promise<Array<Pick<Session, 'id' | 'goal' | 'created_at' | 'updated_at' | 'status'>>> {
  return request('/api/sessions')
}

export function getSession(id: string): Promise<Session> {
  return request<Session>(`/api/sessions/${id}`)
}

export function sendChat(sessionId: string, content: string): Promise<ChatResponse> {
  return request<ChatResponse>(`/api/sessions/${sessionId}/chat`, {
    method: 'POST',
    body: JSON.stringify({ content }),
  })
}

export function getHealth(): Promise<{ ok: boolean; ai_enabled: boolean; model: string | null }> {
  return request('/api/health')
}
