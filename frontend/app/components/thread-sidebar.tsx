"use client";

import { useRef, useState } from "react";
import type { Thread } from "../lib/threads";

type ThreadSidebarProps = {
  threads: Thread[];
  currentThreadId: string | null;
  onSelect: (threadId: string) => void;
  onCreate: () => void;
  onRename: (threadId: string, title: string) => void;
  onDelete: (threadId: string) => void;
};

function formatRelativeTime(iso: string | null): string {
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

export default function ThreadSidebar({
  threads,
  currentThreadId,
  onSelect,
  onCreate,
  onRename,
  onDelete,
}: ThreadSidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  // 防止 Enter 提交后又触发 input 的 onBlur 造成重复提交
  const editingRef = useRef(false);

  const startEdit = (t: Thread) => {
    editingRef.current = true;
    setEditingId(t.thread_id);
    setDraftTitle(t.title ?? "");
  };

  const finishEdit = (commit: boolean) => {
    if (!editingRef.current) return;
    editingRef.current = false;
    if (commit && editingId) {
      onRename(editingId, draftTitle.trim());
    }
    setEditingId(null);
  };

  return (
    <aside className="w-64 shrink-0 border-r border-zinc-200 bg-zinc-50 flex flex-col">
      <div className="p-3 border-b border-zinc-200">
        <button
          onClick={onCreate}
          className="w-full rounded-lg bg-zinc-900 px-3 py-2 text-sm font-medium text-white transition hover:bg-zinc-700"
        >
          + 新建会话
        </button>
      </div>

      <nav className="flex-1 overflow-y-auto p-2 space-y-1">
        {threads.length === 0 && (
          <p className="px-2 py-4 text-center text-sm text-zinc-400">暂无会话</p>
        )}
        {threads.map((t) => {
          const active = t.thread_id === currentThreadId;
          const isEditing = editingId === t.thread_id;
          const isConfirming = confirmingDeleteId === t.thread_id;

          return (
            <div
              key={t.thread_id}
              className={`group rounded-lg px-3 py-2 transition ${
                active ? "bg-zinc-200" : "cursor-pointer hover:bg-zinc-100"
              }`}
              onClick={() => {
                if (!isEditing && !isConfirming) onSelect(t.thread_id);
              }}
            >
              {isEditing ? (
                <input
                  autoFocus
                  value={draftTitle}
                  onChange={(e) => setDraftTitle(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") finishEdit(true);
                    if (e.key === "Escape") finishEdit(false);
                  }}
                  onBlur={() => finishEdit(true)}
                  className="w-full rounded border border-zinc-300 px-1.5 py-0.5 text-sm text-zinc-800"
                  onClick={(e) => e.stopPropagation()}
                  placeholder="输入会话标题"
                />
              ) : (
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium text-zinc-800">
                      {t.title || "新对话"}
                    </div>
                    <div className="text-xs text-zinc-400">
                      {formatRelativeTime(t.updated_at)}
                    </div>
                  </div>

                  {isConfirming ? (
                    <div className="flex shrink-0 items-center gap-1.5">
                      <button
                        className="text-xs text-red-600 hover:underline"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDelete(t.thread_id);
                          setConfirmingDeleteId(null);
                        }}
                      >
                        删除
                      </button>
                      <button
                        className="text-xs text-zinc-500 hover:underline"
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirmingDeleteId(null);
                        }}
                      >
                        取消
                      </button>
                    </div>
                  ) : (
                    <div className="hidden shrink-0 items-center gap-1.5 group-hover:flex">
                      <button
                        aria-label="重命名"
                        className="text-xs text-zinc-500 hover:text-zinc-900"
                        onClick={(e) => {
                          e.stopPropagation();
                          startEdit(t);
                        }}
                      >
                        重命名
                      </button>
                      <button
                        aria-label="删除"
                        className="text-xs text-red-500 hover:text-red-700"
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirmingDeleteId(t.thread_id);
                        }}
                      >
                        删除
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
