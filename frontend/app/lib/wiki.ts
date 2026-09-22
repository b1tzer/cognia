// 个人 Wiki 的类型定义与 API 客户端（走 /api/wiki 代理到 FastAPI 后端）。

export type WikiAuthor = "ai" | "user";

export type WikiPageMeta = {
  page_id: string;
  title: string;
  path: string;
  tags: string[];
  parent_page_id: string | null;
  author: WikiAuthor;
  source_thread_id: string | null;
  source_turns: number[];
  evidence: string[];
  updated_at: string;
  commit_hash: string;
};

export type WikiPage = WikiPageMeta & {
  content_markdown: string;
};

export type WikiRevision = {
  commit_hash: string;
  author: WikiAuthor;
  timestamp: string;
  summary: string;
};

export type WikiUpsertPayload = {
  page_id: string;
  title: string;
  content_markdown: string;
  tags?: string[];
  author?: WikiAuthor;
};

function withUser(path: string, userId: string): string {
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}user_id=${encodeURIComponent(userId)}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`/api/wiki${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!resp.ok) {
    let detail = "";
    try {
      const data = await resp.json();
      detail = data?.detail ?? data?.message ?? "";
    } catch {
      detail = "";
    }
    throw new Error(detail || `wiki 操作失败（HTTP ${resp.status}）`);
  }
  return (await resp.json()) as T;
}

export async function listPages(userId: string): Promise<WikiPageMeta[]> {
  const data = await request<{ pages: WikiPageMeta[] }>(withUser("", userId));
  return data.pages ?? [];
}

export async function getPage(userId: string, pageId: string): Promise<WikiPage> {
  return request<WikiPage>(withUser(`/${pageId}`, userId));
}

export async function createPage(
  userId: string,
  payload: WikiUpsertPayload,
): Promise<WikiPageMeta> {
  return request<WikiPageMeta>(withUser("", userId), {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function updatePage(
  userId: string,
  pageId: string,
  payload: WikiUpsertPayload,
): Promise<WikiPageMeta> {
  return request<WikiPageMeta>(withUser(`/${pageId}`, userId), {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}

export async function deletePage(userId: string, pageId: string): Promise<void> {
  const resp = await fetch(withUser(`/api/wiki/${pageId}`, userId), {
    method: "DELETE",
  });
  if (!resp.ok) {
    throw new Error(`删除页面失败（HTTP ${resp.status}）`);
  }
}

export async function getHistory(
  userId: string,
  pageId: string,
): Promise<WikiRevision[]> {
  const data = await request<{ revisions: WikiRevision[] }>(
    withUser(`/${pageId}/history`, userId),
  );
  return data.revisions ?? [];
}

export async function rollback(
  userId: string,
  pageId: string,
  commitHash: string,
): Promise<WikiPageMeta> {
  return request<WikiPageMeta>(withUser(`/${pageId}/rollback`, userId), {
    method: "POST",
    body: JSON.stringify({ commit_hash: commitHash }),
  });
}

export async function summarize(
  userId: string,
  threadId: string,
  title?: string,
): Promise<{ ok: boolean; page?: WikiPageMeta; reason?: string; message?: string }> {
  return request<{
    ok: boolean;
    page?: WikiPageMeta;
    reason?: string;
    message?: string;
  }>(withUser("/summarize", userId), {
    method: "POST",
    body: JSON.stringify({ thread_id: threadId, title: title ?? null }),
  });
}
