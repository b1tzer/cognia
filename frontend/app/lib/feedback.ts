// 消息反馈（点赞 / 点踩）API 封装。权威数据在服务端 PostgreSQL 的
// message_feedback 表，经 Next.js 代理转发到 FastAPI /feedback 端点。

export type FeedbackValue = "up" | "down";

// 某会话全部反馈：{ messageId: "up" | "down" }
export type FeedbackMap = Record<string, FeedbackValue>;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`/api/feedback${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!resp.ok) {
    throw new Error(`反馈操作失败（HTTP ${resp.status}）`);
  }
  return (await resp.json()) as T;
}

export async function upsertFeedback(
  threadId: string,
  messageId: string,
  userId: string,
  feedback: FeedbackValue,
): Promise<void> {
  await request("", {
    method: "PUT",
    body: JSON.stringify({ thread_id: threadId, message_id: messageId, user_id: userId, feedback }),
  });
}

export async function deleteFeedback(
  threadId: string,
  messageId: string,
  userId: string,
): Promise<void> {
  await request("", {
    method: "DELETE",
    body: JSON.stringify({ thread_id: threadId, message_id: messageId, user_id: userId }),
  });
}

export async function listFeedback(
  threadId: string,
  userId: string,
): Promise<FeedbackMap> {
  const resp = await fetch(
    `/api/feedback?thread_id=${encodeURIComponent(threadId)}&user_id=${encodeURIComponent(userId)}`,
    { cache: "no-store" },
  );
  if (!resp.ok) {
    throw new Error(`读取反馈失败（HTTP ${resp.status}）`);
  }
  return (await resp.json()) as FeedbackMap;
}
