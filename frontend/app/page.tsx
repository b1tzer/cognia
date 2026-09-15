"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CopilotKit,
  useAgent,
  useCopilotKit,
} from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import { Markdown } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
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

function MessageBubble({
  message,
  toolName,
  isActiveReasoning,
}: {
  message: any;
  toolName?: string;
  isActiveReasoning?: boolean;
}) {
  const role = message.role as string;
  if (role === "tool") {
    return (
      <div className="px-4 py-1 text-xs text-zinc-400">
        <span className="rounded bg-zinc-100 px-2 py-1">
          🔧 {toolName || "工具"} 调用完成
        </span>
      </div>
    );
  }
  if (role === "reasoning") {
    const text = typeof message.content === "string" ? message.content : "";
    return (
      <div className="px-4 py-1">
        <details className="text-xs text-zinc-500" open={isActiveReasoning}>
          <summary className="cursor-pointer select-none italic">
            {isActiveReasoning ? "思考中…" : "思考过程"}
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
      {isUser ? (
        <div className="max-w-[75%] whitespace-pre-wrap rounded-2xl bg-zinc-900 px-4 py-2 text-sm text-white">
          {text}
        </div>
      ) : (
        <div className="max-w-[85%] rounded-2xl bg-zinc-100 px-4 py-2 text-sm text-zinc-800">
          <Markdown content={text} />
        </div>
      )}
    </div>
  );
}

// 把 agent.messages 重排为「显示顺序」，并补出工具名称 / reasoning 活跃状态。
//
// 三个目标：
// 1. 工具调用结果（role=tool）应显示在其对应 assistant 工具调用之后、最终回答之前，
//    而不是出现在最终回答下面。根源是 CopilotKit 里同一个 assistant 消息可能同时
//    携带 toolCalls 和 content，直接按数组顺序渲染会让 content 排在 tool 结果前面。
// 2. tool 消息本身没有工具名，需从 assistant.toolCalls[].function.name 映射。
// 3. reasoning 只有在「agent 仍在运行且后面还没有正式回答」时展开，其余默认折叠。
function buildDisplayMessages(messages: any[], isRunning: boolean) {
  // toolCallId -> 工具名
  const toolNames = new Map<string, string>();
  for (const m of messages) {
    if (m.role === "assistant" && Array.isArray(m.toolCalls)) {
      for (const tc of m.toolCalls) {
        if (tc?.id && tc?.function?.name) {
          toolNames.set(tc.id, tc.function.name);
        }
      }
    }
  }

  // toolCallId -> tool 结果消息
  const toolResultByCallId = new Map<string, any>();
  for (const m of messages) {
    if (m.role === "tool" && m.toolCallId) {
      toolResultByCallId.set(m.toolCallId, m);
    }
  }

  const display: Array<{
    message: any;
    toolName?: string;
    isActiveReasoning?: boolean;
  }> = [];
  const consumedToolCallIds = new Set<string>();

  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];

    // tool 结果不在此处输出，由对应的 assistant 工具调用消息统一插入，
    // 以保证它位于最终回答之前。
    if (m.role === "tool") continue;

    if (
      m.role === "assistant" &&
      Array.isArray(m.toolCalls) &&
      m.toolCalls.length > 0
    ) {
      // 先插入对应的工具调用结果
      for (const tc of m.toolCalls) {
        const result = toolResultByCallId.get(tc.id);
        if (result && !consumedToolCallIds.has(tc.id)) {
          display.push({
            message: result,
            toolName: toolNames.get(tc.id),
          });
          consumedToolCallIds.add(tc.id);
        }
      }

      // 再输出该 assistant 的正式回答（如有文本）
      const content =
        typeof m.content === "string" && m.content.trim() ? m.content : "";
      if (content) {
        display.push({ message: { ...m, content } });
      }
      // 纯工具调用（无文本）不额外输出空气泡
    } else if (m.role === "reasoning") {
      // 该 reasoning 之后是否已经出现 assistant 正式回答
      const hasAnswerAfter = messages.slice(i + 1).some(
        (x) =>
          x.role === "assistant" &&
          typeof x.content === "string" &&
          x.content.trim(),
      );
      display.push({
        message: m,
        isActiveReasoning: !hasAnswerAfter && isRunning,
      });
    } else {
      display.push({ message: m });
    }
  }

  // 兜底：没有对应 assistant 工具调用的 tool 结果（历史数据异常等）
  for (const m of messages) {
    if (
      m.role === "tool" &&
      m.toolCallId &&
      !consumedToolCallIds.has(m.toolCallId)
    ) {
      display.push({
        message: m,
        toolName: toolNames.get(m.toolCallId),
      });
      consumedToolCallIds.add(m.toolCallId);
    }
  }

  return display;
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
  const displayMessages = useMemo(
    () => buildDisplayMessages(messages, agent?.isRunning ?? false),
    [messages, agent?.isRunning],
  );

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
            displayMessages.map((item) => (
              <MessageBubble
                key={item.message.id ?? crypto.randomUUID()}
                message={item.message}
                toolName={item.toolName}
                isActiveReasoning={item.isActiveReasoning}
              />
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
