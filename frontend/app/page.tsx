"use client";

import { useCallback, useEffect, useState } from "react";
import { CopilotKit, CopilotChat } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import ThreadSidebar from "./components/thread-sidebar";
import {
  createThread,
  deleteThread,
  listThreads,
  renameThread,
  type Thread,
} from "./lib/threads";

// localStorage 只保存「当前选中的 threadId」。会话列表的权威数据在服务端
// PostgreSQL（GET /threads），不在本地。刷新后据此恢复当前会话。
const THREAD_ID_KEY = "cognia:threadId";

export default function Home() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  const persistCurrentThreadId = useCallback((id: string) => {
    window.localStorage.setItem(THREAD_ID_KEY, id);
  }, []);

  const refreshThreads = useCallback(async () => {
    const list = await listThreads();
    setThreads(list);
    return list;
  }, []);

  // 初始化：拉取服务端会话列表，并恢复 currentThreadId（优先 localStorage，
  // 否则选最新一条；列表为空则先创建一个新会话）。
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const saved = window.localStorage.getItem(THREAD_ID_KEY);
        let list = await refreshThreads();

        if (list.length === 0) {
          const created = await createThread();
          list = [created];
          setThreads(list);
        }

        const exists = list.some((t) => t.thread_id === saved);
        const selected = exists && saved ? saved : list[0].thread_id;
        if (!cancelled) {
          setCurrentThreadId(selected);
          persistCurrentThreadId(selected);
          setReady(true);
        }
      } catch (err) {
        console.error("初始化会话失败", err);
        if (!cancelled) {
          // 降级兜底：Postgres 不可用（会话管理 503）时，仍保留原有单会话
          // 聊天能力——读 localStorage 或临时生成一个 threadId。
          const fallback =
            window.localStorage.getItem(THREAD_ID_KEY) || crypto.randomUUID();
          setCurrentThreadId(fallback);
          persistCurrentThreadId(fallback);
          setReady(true);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshThreads, persistCurrentThreadId]);

  const handleSelect = (threadId: string) => {
    setCurrentThreadId(threadId);
    persistCurrentThreadId(threadId);
  };

  const handleCreate = async () => {
    const created = await createThread();
    setThreads((prev) => [created, ...prev]);
    handleSelect(created.thread_id);
  };

  const handleRename = async (threadId: string, title: string) => {
    const updated = await renameThread(threadId, title);
    setThreads((prev) =>
      prev.map((t) => (t.thread_id === threadId ? updated : t)),
    );
  };

  const handleDelete = async (threadId: string) => {
    await deleteThread(threadId);
    let list = await refreshThreads();

    // 删除的是当前会话：自动选择最新一条；没有会话则新建一个。
    if (threadId === currentThreadId) {
      if (list.length === 0) {
        const created = await createThread();
        list = [created];
      }
      handleSelect(list[0].thread_id);
    }
    setThreads(list);
  };

  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <div className="flex h-screen w-full">
        <ThreadSidebar
          threads={threads}
          currentThreadId={currentThreadId}
          onSelect={handleSelect}
          onCreate={handleCreate}
          onRename={handleRename}
          onDelete={handleDelete}
        />

        <div className="flex min-w-0 flex-1 flex-col">
          <header className="border-b border-zinc-200 bg-white px-6 py-3">
            <h1 className="text-lg font-semibold text-zinc-900">
              Cognia · AI 学习教练
            </h1>
            <p className="text-sm text-zinc-500">
              主动发现认知盲区，动态引导掌握知识点
            </p>
          </header>
          <div className="min-h-0 flex-1">
            {ready && currentThreadId ? (
              <CopilotChat
                key={currentThreadId}
                threadId={currentThreadId}
                labels={{
                  title: "Cognia 学习教练",
                  initial:
                    "你好！我是 Cognia，你的 AI 学习教练。\n请告诉我你想学什么（例如「Spring AOP」），我会先了解你的基础，再针对性教学。",
                }}
              />
            ) : (
              <div className="h-full" />
            )}
          </div>
        </div>
      </div>
    </CopilotKit>
  );
}

