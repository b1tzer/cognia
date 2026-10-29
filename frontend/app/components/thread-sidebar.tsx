"use client";

import { useRef, useState } from "react";
import {
  BookOpen,
  Network,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Plus,
  Trash2,
} from "lucide-react";
import type { Thread } from "../lib/threads";

type ThreadSidebarProps = {
  threads: Thread[];
  currentThreadId: string | null;
  view: "chat" | "map" | "wiki";
  onSelect: (threadId: string) => void;
  onCreate: () => void;
  onRename: (threadId: string, title: string) => void;
  onDelete: (threadId: string) => void;
  onNavigate: (view: "chat" | "map" | "wiki") => void;
};

// 侧边栏功能导航项：对齐 ChatGPT 的「功能入口」信息架构，
// 把原先顶部的三个视图 tab 下沉到侧边栏，主区不再放导航。
const NAV_ITEMS = [
  { key: "chat", label: "新建对话", icon: Plus },
  { key: "map", label: "知识版图", icon: Network },
  { key: "wiki", label: "Wiki", icon: BookOpen },
] as const;

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
  view,
  onSelect,
  onCreate,
  onRename,
  onDelete,
  onNavigate,
}: ThreadSidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  // 侧边栏折叠状态：折叠后只留一个「展开」按钮，主区自动扩展。
  const [collapsed, setCollapsed] = useState(false);
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

  // 折叠态：只渲染一个「展开」按钮，主区自动扩展。
  if (collapsed) {
    return (
      <aside className="flex shrink-0 flex-col border-r border-line bg-surface">
        <button
          onClick={() => setCollapsed(false)}
          aria-label="展开侧边栏"
          title="展开侧边栏"
          className="m-2 inline-flex items-center justify-center rounded-lg p-2 text-muted transition hover:bg-surface-muted hover:text-foreground"
        >
          <PanelLeftOpen className="h-5 w-5" aria-hidden />
        </button>
      </aside>
    );
  }

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-line bg-surface">
      {/* 品牌区 + 折叠按钮 */}
      <div className="flex items-center justify-between px-3 py-3">
        <h1 className="text-base font-semibold text-foreground">Cognia</h1>
        <button
          onClick={() => setCollapsed(true)}
          aria-label="收起侧边栏"
          title="收起侧边栏"
          className="inline-flex items-center justify-center rounded-lg p-1.5 text-muted transition hover:bg-surface-muted hover:text-foreground"
        >
          <PanelLeftClose className="h-5 w-5" aria-hidden />
        </button>
      </div>

      {/* 功能导航：新建对话 / 知识版图 / Wiki */}
      <nav aria-label="功能导航" className="px-3 py-2">
        <ul className="space-y-1">
          {NAV_ITEMS.map(({ key, label, icon: Icon }) => (
            <li key={key}>
              <button
                onClick={() => (key === "chat" ? onCreate() : onNavigate(key))}
                className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm transition ${
                  view === key && key !== "chat"
                    ? "bg-surface-muted font-medium text-foreground"
                    : "text-muted hover:bg-surface-muted hover:text-foreground"
                }`}
              >
                <Icon className="h-4 w-4" aria-hidden />
                {label}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      {/* 会话列表 */}
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto p-2">
        <p className="px-2 pb-1 text-xs font-medium text-muted">会话</p>
        {threads.length === 0 && (
          <p className="px-2 py-4 text-center text-sm text-muted">暂无会话</p>
        )}
        <ul className="space-y-1">
          {threads.map((t) => {
            const active = t.thread_id === currentThreadId;
            const isEditing = editingId === t.thread_id;
            const isConfirming = confirmingDeleteId === t.thread_id;

            return (
              <li
                key={t.thread_id}
                className={`group rounded-lg px-3 py-2 transition ${
                  active
                    ? "bg-surface-muted"
                    : "cursor-pointer hover:bg-surface-muted"
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
                    className="w-full rounded border border-line px-1.5 py-0.5 text-sm text-foreground"
                    onClick={(e) => e.stopPropagation()}
                    placeholder="输入会话标题"
                  />
                ) : (
                  <div className="flex items-center justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-foreground">
                        {t.title || "新对话"}
                      </div>
                      <div className="text-xs text-muted">
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
                          className="text-xs text-muted hover:underline"
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
                          title="重命名"
                          className="inline-flex items-center justify-center rounded p-1 text-muted transition hover:bg-surface-muted hover:text-foreground"
                          onClick={(e) => {
                            e.stopPropagation();
                            startEdit(t);
                          }}
                        >
                          <Pencil className="h-3.5 w-3.5" aria-hidden />
                        </button>
                        <button
                          aria-label="删除"
                          title="删除"
                          className="inline-flex items-center justify-center rounded p-1 text-muted transition hover:bg-surface-muted hover:text-red-600"
                          onClick={(e) => {
                            e.stopPropagation();
                            setConfirmingDeleteId(t.thread_id);
                          }}
                        >
                          <Trash2 className="h-3.5 w-3.5" aria-hidden />
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    </aside>
  );
}
