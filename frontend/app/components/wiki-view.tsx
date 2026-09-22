"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Markdown } from "@copilotkit/react-ui";
import {
  createPage,
  deletePage,
  getHistory,
  getPage,
  listPages,
  rollback,
  summarize,
  updatePage,
  type WikiPage,
  type WikiPageMeta,
  type WikiRevision,
} from "../lib/wiki";
import { getOrCreateUserId } from "../lib/user-id";

type WikiViewProps = {
  currentThreadId: string | null;
};

type Mode = "view" | "edit";

function formatTime(iso: string | null): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const diffMs = Date.now() - date.getTime();
  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days} 天前`;
  return date.toLocaleDateString("zh-CN");
}

function newSlug(): string {
  return `wiki-${Date.now().toString(36)}`;
}

export default function WikiView({ currentThreadId }: WikiViewProps) {
  const [userId] = useState(() => getOrCreateUserId());
  const [pages, setPages] = useState<WikiPageMeta[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [page, setPage] = useState<WikiPage | null>(null);
  const [mode, setMode] = useState<Mode>("view");
  const [draftTitle, setDraftTitle] = useState("");
  const [draftContent, setDraftContent] = useState("");
  const [draftIsNew, setDraftIsNew] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [revisions, setRevisions] = useState<WikiRevision[]>([]);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const loadPages = useCallback(async () => {
    try {
      const list = await listPages(userId);
      setPages(list);
    } catch (err) {
      console.error("加载 wiki 列表失败", err);
      setError("加载 wiki 列表失败，请稍后重试");
    }
  }, [userId]);

  useEffect(() => {
    void loadPages();
  }, [loadPages]);

  const selectPage = useCallback(
    async (pageId: string) => {
      setLoading(true);
      setError(null);
      setMode("view");
      setShowHistory(false);
      setSelectedId(pageId);
      try {
        const p = await getPage(userId, pageId);
        setPage(p);
      } catch (err) {
        console.error("读取 wiki 页面失败", err);
        setError("读取页面失败，请稍后重试");
      } finally {
        setLoading(false);
      }
    },
    [userId],
  );

  const startCreate = useCallback(() => {
    setMode("edit");
    setDraftIsNew(true);
    setDraftTitle("");
    setDraftContent("");
    setSelectedId(null);
    setPage(null);
    setShowHistory(false);
    setError(null);
  }, []);

  const startEdit = useCallback(() => {
    if (!page) return;
    setMode("edit");
    setDraftIsNew(false);
    setDraftTitle(page.title);
    setDraftContent(page.content_markdown);
    setShowHistory(false);
    setError(null);
  }, [page]);

  const saveDraft = useCallback(async () => {
    const title = draftTitle.trim();
    const content = draftContent.trim();
    if (!title) {
      setError("标题不能为空");
      return;
    }
    if (!content) {
      setError("正文不能为空");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      if (draftIsNew) {
        const meta = await createPage(userId, {
          page_id: newSlug(),
          title,
          content_markdown: content,
          author: "user",
        });
        setPages((prev) => [meta, ...prev]);
        setSelectedId(meta.page_id);
        setPage({ ...meta, content_markdown: content });
      } else if (selectedId) {
        const meta = await updatePage(userId, selectedId, {
          page_id: selectedId,
          title,
          content_markdown: content,
          author: "user",
        });
        setPages((prev) =>
          prev.map((p) => (p.page_id === selectedId ? meta : p)),
        );
        setPage({ ...meta, content_markdown: content });
      }
      setMode("view");
    } catch (err) {
      console.error("保存 wiki 页面失败", err);
      setError(err instanceof Error ? err.message : "保存失败，请稍后重试");
    } finally {
      setLoading(false);
    }
  }, [draftTitle, draftContent, draftIsNew, selectedId, userId]);

  const doDelete = useCallback(async () => {
    if (!selectedId) return;
    try {
      await deletePage(userId, selectedId);
      setPages((prev) => prev.filter((p) => p.page_id !== selectedId));
      setSelectedId(null);
      setPage(null);
      setMode("view");
      setShowHistory(false);
    } catch (err) {
      console.error("删除 wiki 页面失败", err);
      setError(err instanceof Error ? err.message : "删除失败，请稍后重试");
    }
  }, [selectedId, userId]);

  const loadHistory = useCallback(async () => {
    if (!selectedId) return;
    try {
      const revs = await getHistory(userId, selectedId);
      setRevisions(revs);
      setShowHistory(true);
    } catch (err) {
      console.error("读取版本历史失败", err);
      setError(err instanceof Error ? err.message : "读取版本历史失败");
    }
  }, [selectedId, userId]);

  const doRollback = useCallback(
    async (commitHash: string) => {
      if (!selectedId) return;
      try {
        await rollback(userId, selectedId, commitHash);
        setShowHistory(false);
        await selectPage(selectedId);
        await loadPages();
      } catch (err) {
        console.error("回滚失败", err);
        setError(err instanceof Error ? err.message : "回滚失败，请稍后重试");
      }
    },
    [selectedId, userId, selectPage, loadPages],
  );

  const doSummarize = useCallback(async () => {
    if (!currentThreadId) {
      setError("当前没有可总结的会话");
      return;
    }
    setGenerating(true);
    setNotice(null);
    setError(null);
    try {
      const result = await summarize(userId, currentThreadId);
      if (result.ok && result.page) {
        await loadPages();
        await selectPage(result.page.page_id);
        setNotice("已生成 wiki 草稿");
      } else {
        setNotice(result.message || "本次会话无可总结内容");
      }
    } catch (err) {
      console.error("生成 wiki 失败", err);
      setError(err instanceof Error ? err.message : "生成失败，请稍后重试");
    } finally {
      setGenerating(false);
    }
  }, [currentThreadId, userId, loadPages, selectPage]);

  const displayContent = useMemo(() => {
    if (mode === "edit") return draftContent;
    return page?.content_markdown ?? "";
  }, [mode, draftContent, page]);

  return (
    <div className="flex h-full min-h-0 flex-1">
      {/* 左侧：页面列表 + 工具栏 */}
      <aside className="flex w-64 shrink-0 flex-col border-r border-line bg-surface-muted">
        <div className="space-y-2 border-b border-line p-3">
          <button
            onClick={startCreate}
            className="w-full rounded-lg bg-zinc-900 px-3 py-2 text-sm font-medium text-white transition hover:bg-zinc-700"
          >
            + 新建页面
          </button>
          <button
            onClick={() => void doSummarize()}
            disabled={!currentThreadId || generating}
            className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-foreground transition hover:bg-surface-muted disabled:opacity-40"
          >
            {generating ? "生成中…" : "从当前会话生成"}
          </button>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto p-2">
          {pages.length === 0 && (
            <p className="px-2 py-4 text-center text-sm text-muted">暂无 wiki 页面</p>
          )}
          {pages.map((p) => {
            const active = p.page_id === selectedId;
            return (
              <div
                key={p.page_id}
                onClick={() => void selectPage(p.page_id)}
                className={`group rounded-lg px-3 py-2 transition ${
                  active ? "bg-zinc-200" : "cursor-pointer hover:bg-surface"
                }`}
              >
                <div className="truncate text-sm font-medium text-foreground">
                  {p.title || "未命名"}
                </div>
                <div className="flex items-center gap-1.5 text-xs text-muted">
                  <span>{p.author === "ai" ? "AI 生成" : "手写"}</span>
                  <span>·</span>
                  <span>{formatTime(p.updated_at)}</span>
                </div>
              </div>
            );
          })}
        </nav>
      </aside>

      {/* 右侧：详情 / 编辑 / 历史 */}
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-between border-b border-line bg-surface px-6 py-3">
          <div className="min-w-0">
            <h2 className="truncate text-lg font-semibold text-foreground">
              {mode === "edit"
                ? draftIsNew
                  ? "新建页面"
                  : "编辑页面"
                : page?.title || "Wiki"}
            </h2>
            {page && mode === "view" && (
              <p className="text-xs text-muted">
                {page.author === "ai" ? "AI 生成" : "手写"}
                {page.source_thread_id ? " · 含溯源" : ""} · 更新于{" "}
                {formatTime(page.updated_at)}
              </p>
            )}
          </div>

          <div className="flex shrink-0 items-center gap-2">
            {mode === "view" && page && (
              <>
                <button
                  onClick={startEdit}
                  className="rounded-lg border border-line px-3 py-1.5 text-sm text-foreground transition hover:bg-surface-muted"
                >
                  编辑
                </button>
                <button
                  onClick={() => void loadHistory()}
                  className="rounded-lg border border-line px-3 py-1.5 text-sm text-foreground transition hover:bg-surface-muted"
                >
                  历史
                </button>
                <button
                  onClick={() => void doDelete()}
                  className="rounded-lg border border-line px-3 py-1.5 text-sm text-red-600 transition hover:bg-surface-muted"
                >
                  删除
                </button>
              </>
            )}
            {mode === "edit" && (
              <>
                <button
                  onClick={() => {
                    setMode("view");
                    setShowHistory(false);
                    if (draftIsNew && !page) {
                      setSelectedId(null);
                    }
                  }}
                  className="rounded-lg border border-line px-3 py-1.5 text-sm text-foreground transition hover:bg-surface-muted"
                >
                  取消
                </button>
                <button
                  onClick={() => void saveDraft()}
                  disabled={loading}
                  className="rounded-lg bg-ink px-4 py-1.5 text-sm font-medium text-white transition hover:bg-ink/90 disabled:opacity-40"
                >
                  {loading ? "保存中…" : "保存"}
                </button>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="border-b border-line bg-surface px-6 py-2 text-sm text-red-600">
            {error}
          </div>
        )}
        {notice && (
          <div className="border-b border-line bg-surface px-6 py-2 text-sm text-muted">
            {notice}
          </div>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto p-6">
          {showHistory ? (
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-foreground">版本历史</h3>
              {revisions.length === 0 ? (
                <p className="text-sm text-muted">暂无历史版本</p>
              ) : (
                revisions.map((rev, idx) => (
                  <div
                    key={rev.commit_hash}
                    className="flex items-center justify-between rounded-lg border border-line bg-surface px-4 py-2"
                  >
                    <div className="min-w-0">
                      <div className="text-sm text-foreground">
                        {rev.summary || `版本 ${revisions.length - idx}`}
                        {idx === 0 && (
                          <span className="ml-2 rounded bg-surface-muted px-1.5 py-0.5 text-xs text-muted">
                            当前
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-muted">
                        {rev.author === "ai" ? "AI 生成" : "手写"} ·{" "}
                        {formatTime(rev.timestamp)} ·{" "}
                        <code className="font-mono">{rev.commit_hash.slice(0, 8)}</code>
                      </div>
                    </div>
                    {idx > 0 && (
                      <button
                        onClick={() => void doRollback(rev.commit_hash)}
                        className="rounded-lg border border-line px-3 py-1 text-xs text-foreground transition hover:bg-surface-muted"
                      >
                        回滚到此版本
                      </button>
                    )}
                  </div>
                ))
              )}
            </div>
          ) : loading && !page ? (
            <div className="flex h-full items-center justify-center text-sm text-muted">
              加载中…
            </div>
          ) : !page && mode === "view" ? (
            <div className="flex h-full items-center justify-center text-sm text-muted">
              选择或新建一个 wiki 页面
            </div>
          ) : mode === "edit" ? (
            <div className="grid h-full grid-cols-2 gap-4">
              <div className="flex flex-col">
                <input
                  value={draftTitle}
                  onChange={(e) => setDraftTitle(e.target.value)}
                  placeholder="标题"
                  className="mb-3 rounded-lg border border-line bg-surface px-4 py-2 text-sm text-foreground outline-none focus:border-accent"
                />
                <textarea
                  value={draftContent}
                  onChange={(e) => setDraftContent(e.target.value)}
                  placeholder="用 Markdown 撰写正文…"
                  className="min-h-0 flex-1 resize-none rounded-lg border border-line bg-surface px-4 py-3 font-mono text-sm text-foreground outline-none focus:border-accent"
                />
              </div>
              <div className="min-h-0 overflow-y-auto rounded-lg border border-line bg-surface px-4 py-3">
                <div className="prose-sm max-w-none text-foreground">
                  <Markdown content={displayContent || "*预览区：正文将在此渲染*"} />
                </div>
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl">
              <div className="text-foreground">
                <Markdown content={displayContent} />
              </div>

              {page?.evidence && page.evidence.length > 0 && (
                <div className="mt-6 rounded-lg border border-line bg-surface-muted p-4">
                  <h4 className="mb-2 text-sm font-semibold text-foreground">
                    溯源 · 引用原话
                  </h4>
                  <ul className="space-y-1 text-sm text-muted">
                    {page.evidence.map((e, i) => (
                      <li key={i}>“{e}”</li>
                    ))}
                  </ul>
                  {page.source_thread_id && (
                    <p className="mt-2 text-xs text-muted">
                      来源会话：{page.source_thread_id}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
