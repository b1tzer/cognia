import type { ChatResponse, Session, Message, LLMTrace } from './types'

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

export function confirmGoal(sessionId: string, goal: string): Promise<Session> {
  return request<Session>(`/api/sessions/${sessionId}/confirm-goal`, {
    method: 'POST',
    body: JSON.stringify({ goal }),
  })
}

export function updateGoal(sessionId: string, goal: string): Promise<Session> {
  return request<Session>(`/api/sessions/${sessionId}/goal`, {
    method: 'PUT',
    body: JSON.stringify({ goal }),
  })
}

export function sendChat(sessionId: string, content: string): Promise<ChatResponse> {
  return request<ChatResponse>(`/api/sessions/${sessionId}/chat`, {
    method: 'POST',
    body: JSON.stringify({ content }),
  })
}

export interface StreamCallbacks {
  onToken: (content: string) => void
  onTrace: (step: LLMTrace, index: number) => void
  onDone: (data: ChatResponse) => void
  onError: (err: Error) => void
}

/**
 * 流式发送消息：消费后端 SSE 接口 /api/sessions/{sid}/chat/stream。
 * SSE 协议：逐行 "data: {json}"，事件类型 type=token（文本增量）/ done（完整 payload）。
 * 用 ReadableStream + TextDecoder 逐块读取，按行累积解析（兼容 chunk 含多行或不完整行）。
 */
export async function sendChatStream(
  sessionId: string,
  content: string,
  cb: StreamCallbacks,
): Promise<void> {
  const res = await fetch(`/api/sessions/${sessionId}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail || `请求失败 (${res.status})`)
  }
  if (!res.body) {
    throw new Error('当前浏览器不支持流式读取')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx).trim()
        buffer = buffer.slice(idx + 1)
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (!payload) continue
        try {
          const evt = JSON.parse(payload)
          if (evt.type === 'token' && typeof evt.content === 'string') {
            cb.onToken(evt.content)
          } else if (evt.type === 'trace' && evt.step) {
            cb.onTrace(evt.step as LLMTrace, Number(evt.index ?? 0))
          } else if (evt.type === 'done' && evt.data) {
            cb.onDone(evt.data as ChatResponse)
          }
        } catch {
          // 忽略无法解析的行，避免单行坏数据中断整个流
        }
      }
    }
  } catch (e) {
    cb.onError(e as Error)
    throw e
  } finally {
    reader.releaseLock()
  }
}

export function deleteSession(sessionId: string): Promise<{ ok: boolean }> {
  return request(`/api/sessions/${sessionId}`, { method: 'DELETE' })
}

export function deleteMessage(sessionId: string, index: number): Promise<{ ok: boolean; messages: Message[] }> {
  return request(`/api/sessions/${sessionId}/messages/${index}`, { method: 'DELETE' })
}

export function updateMessage(sessionId: string, index: number, content: string): Promise<{ ok: boolean; messages: Message[] }> {
  return request(`/api/sessions/${sessionId}/messages/${index}`, {
    method: 'PUT',
    body: JSON.stringify({ content }),
  })
}

export function regenerateMessage(sessionId: string, index: number): Promise<Session> {
  return request<Session>(`/api/sessions/${sessionId}/messages/${index}/regenerate`, {
    method: 'POST',
  })
}

export function getHealth(): Promise<{ ok: boolean; ai_enabled: boolean; model: string | null }> {
  return request('/api/health')
}
