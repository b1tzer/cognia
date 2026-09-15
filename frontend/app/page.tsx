"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  CopilotKit,
  useAgent,
  useCopilotKit,
} from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import ThreadSidebar from "./components/thread-sidebar";
import {
  createThread,
  deleteThread,
  getThreadMessages,
  listThreads,
  renameThread,
  type Thread,
} from "./lib/threads";

// localStorage 只保存「当前选中的 threadId」。会话列表与历史消息的权威数据
// 都在服务端 PostgreSQL；localStorage 不保存线程列表。
const THREAD_ID_KEY = "cognia:threadId";

function MessageBubble({ message }: { message: any }) {
  const role = message.role as string;
  if (role === "tool") {
    return (
      <div className="px-4 py-1 text-xs text-zinc-400">
        <span className="rounded bg-zinc-100 px-2 py-1">🔧 工具调用完成</span>
      </div>
    );
  }
  if (role === "reasoning") {
    const text = typeof message.content === "string" ? message.content : "";
    return (
      <div className="px-4 py-1">
        <details className="text-xs text-zinc-500" open>
          <summary className="cursor-pointer select-none italic">
            思考过程
          </summary>
          <div className="mt-1 whitespace-pre-wrap rounded-lg bg-zinc-50 p-2 text-zinc-500">
            {text || "思考中…"}
          </div>
        </details>
      </div>
    );
  }
  if (role !== "user" && role !== "assistant") return null;

  const isUser = role === "user";
  const text = typeof message.content === "string" ? message.content : "";
  if (!text) return null;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} px-4 py-2`}>
      <div
        className={`max-w-[75%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm ${
          isUser ? "bg-zinc-900 text-white" : "bg-zinc-100 text-zinc-800"
        }`}
      >
        {text}
      </div>
    </div>
  );
}

// 聊天主体：必须在 <CopilotKit> 内部，才能使用 useAgent / useCopilotKit。
function ChatApp() {
  const { agent, isReady } = useAgent({ agentId: "default" });
  const { copilotkit } = useCopilotKit();

  const [threads, setThreads] = useState<Thread[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const persistCurrentThreadId = useCallback((id: string) => {
    window.localStorage.setItem(THREAD_ID_KEY, id);
  }, []);

  const refreshThreads = useCallback(async () => {
    const list = await listThreads();
    setThreads(list);
    return list;
  }, []);

  const switchThread = useCallback(
    async (threadId: string) => {
      setCurrentThreadId(threadId);
      persistCurrentThreadId(threadId);
      if (!agent) return;
      const history = await getThreadMessages(threadId);
      agent.threadId = threadId;
      agent.setMessages(history);
    },
    [agent, persistCurrentThreadId],
  );

  // 初始化：拉线程列表 + 恢复当前线程。
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
        if (!cancelled) setReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshThreads, persistCurrentThreadId]);

  // 就绪且 agent 可用时，加载当前会话历史。
  useEffect(() => {
    if (!ready || !currentThreadId || !agent) return;
    let cancelled = false;
    (async () => {
      const history = await getThreadMessages(currentThreadId);
      if (cancelled) return;
      agent.threadId = currentThreadId;
      agent.setMessages(history);
    })();
    return () => {
      cancelled = true;
    };
  }, [ready, currentThreadId, agent]);

  // 自动滚动到底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [agent?.messages.length]);

  const handleSelect = (threadId: string) => {
    void switchThread(threadId);
  };

  const handleCreate = async () => {
    const created = await createThread();
    setThreads((prev) => [created, ...prev]);
    await switchThread(created.thread_id);
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
    if (threadId === currentThreadId) {
      if (list.length === 0) {
        const created = await createThread();
        list = [created];
      }
      await switchThread(list[0].thread_id);
    }
    setThreads(list);
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || !agent || sending) return;
    setInput("");
    setSending(true);
    try {
      agent.addMessage({
        id: crypto.randomUUID(),
        role: "user",
        content: text,
      });
      await copilotkit.runAgent({ agent });
    } catch (err) {
      console.error("发送失败", err);
    } finally {
      setSending(false);
    }
  };

  const messages = agent?.messages ?? [];

  return (
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

        <div className="min-h-0 flex-1 overflow-y-auto bg-white">
          {!isReady || !ready ? (
            <div className="flex h-full items-center justify-center text-sm text-zinc-400">
              加载中…
            </div>
          ) : messages.length === 0 ? (
            <div className="flex h-full items-center justify-center text-sm text-zinc-400">
              开始一段新对话吧
            </div>
          ) : (
            messages.map((m: any) => (
              <MessageBubble key={m.id ?? crypto.randomUUID()} message={m} />
            ))
          )}
          <div ref={bottomRef} />
        </div>

        <div className="border-t border-zinc-200 bg-white p-4">
          <div className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void handleSend();
                }
              }}
              placeholder="输入消息，Enter 发送，Shift+Enter 换行"
              className="flex-1 rounded-xl border border-zinc-300 px-4 py-2 text-sm text-zinc-800 outline-none focus:border-zinc-500"
            />
            <button
              onClick={() => void handleSend()}
              disabled={sending || !input.trim()}
              className="rounded-xl bg-zinc-900 px-5 py-2 text-sm font-medium text-white transition hover:bg-zinc-700 disabled:opacity-40"
            >
              {sending ? "发送中…" : "发送"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <ChatApp />
    </CopilotKit>
  );
}
