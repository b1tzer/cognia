import type { Message } from "@copilotkit/react-core/v2";

export type Thread = {
  thread_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`/api/threads${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!resp.ok) {
    throw new Error(`会话操作失败（HTTP ${resp.status}）`);
  }
  return (await resp.json()) as T;
}

export async function listThreads(): Promise<Thread[]> {
  return request<Thread[]>("");
}

export async function createThread(): Promise<Thread> {
  return request<Thread>("", { method: "POST" });
}

export async function renameThread(threadId: string, title: string): Promise<Thread> {
  return request<Thread>(`/${threadId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export async function deleteThread(threadId: string): Promise<void> {
  const resp = await fetch(`/api/threads/${threadId}`, { method: "DELETE" });
  if (!resp.ok) {
    throw new Error(`删除会话失败（HTTP ${resp.status}）`);
  }
}

export async function getThreadMessages(
  threadId: string,
): Promise<Message[]> {
  const resp = await fetch(`/api/threads/${threadId}/messages`);
  if (!resp.ok) {
    throw new Error(`读取会话消息失败（HTTP ${resp.status}）`);
  }
  const data = (await resp.json()) as { messages: Message[] };
  return data.messages ?? [];
}
