"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CopilotKit,
  useAgent,
  useCopilotKit,
  useRenderToolCall,
} from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import { Markdown } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import ThreadSidebar from "./components/thread-sidebar";
import KnowledgeMap, {
  type KnowledgeMapData,
  type KnowledgeMapGoal,
} from "./components/knowledge-map";
import {
  cogniaToolRenderers,
  useCogniaFrontendTools,
} from "./components/cognia-frontend-tools";
import {
  createThread,
  deleteThread,
  getThreadMessages,
  listThreads,
  renameThread,
  type Thread,
} from "./lib/threads";
import { getOrCreateUserId } from "./lib/user-id";

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

  // 纯 JSON 内容（如模型复述的知识点列表）用 ```json 代码块包裹，
  // 让 Markdown 以代码块形式高亮展示，而不是折叠/消失。
  const markdown = !isUser && isJsonBlob(text) ? `\`\`\`json\n${text}\n\`\`\`` : text;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} px-4 py-2`}>
      {isUser ? (
        <div className="max-w-[75%] whitespace-pre-wrap rounded-2xl bg-zinc-900 px-4 py-2 text-sm text-white">
          {text}
        </div>
      ) : (
        <div className="max-w-[85%] rounded-2xl bg-zinc-100 px-4 py-2 text-sm text-zinc-800">
          <Markdown content={markdown} />
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

// 判断文本是否为「纯 JSON」（数组或对象）。若最终回答里出现纯 JSON，应当用
// Markdown 代码块（```json）高亮展示，而不是折叠/丢弃。
function isJsonBlob(text: string): boolean {
  const t = text.trim();
  if (!(t.startsWith("[") || t.startsWith("{"))) return false;
  try {
    JSON.parse(t);
    return true;
  } catch {
    return false;
  }
}

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
    toolCall?: any;
    toolMessage?: any;
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
        if (consumedToolCallIds.has(tc.id)) continue;
        const result = toolResultByCallId.get(tc.id);
        display.push({
          message: result,
          toolName: toolNames.get(tc.id),
          toolCall: tc,
          toolMessage: result,
        });
        consumedToolCallIds.add(tc.id);
      }

      // 带工具调用的 assistant 消息，其 content 是「过渡思考」而非最终回答，
      // 折叠进思考过程框，避免被渲染成正式回答气泡。
      const content =
        typeof m.content === "string" && m.content.trim() ? m.content.trim() : "";
      if (content) {
        display.push({
          message: {
            id: `${m.id || `transition-${i}`}-transition`,
            role: "reasoning",
            content,
          },
          isActiveReasoning: false,
        });
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
  // 注册 Cognia 前端交互工具（三），并获取 Generative UI 渲染函数（五）。
  useCogniaFrontendTools();
  const renderToolCall = useRenderToolCall();

  const [threads, setThreads] = useState<Thread[]>([]);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [loadingThread, setLoadingThread] = useState(false);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  // 视图切换：chat（对话）/ map（知识版图）
  const [view, setView] = useState<"chat" | "map">("chat");
  const [mapData, setMapData] = useState<KnowledgeMapGoal[]>([]);
  const [mapLoading, setMapLoading] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  // 滚动容器 ref 与「用户是否手动上滚」状态：用于在思考中/回答流式输出时
  // 自动跟随滚动，且不打断用户主动向上翻阅。
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const isUserScrollUpRef = useRef(false);
  const isProgrammaticScrollRef = useRef(false);

  const persistCurrentThreadId = useCallback((id: string) => {
    window.localStorage.setItem(THREAD_ID_KEY, id);
  }, []);

  const refreshThreads = useCallback(async () => {
    const list = await listThreads();
    setThreads(list);
    return list;
  }, []);

  const switchThread = useCallback(
    (threadId: string) => {
      // 只做「选中会话」的状态切换；历史消息的统一加载交给下方
      // 监听 currentThreadId 的 useEffect，避免一次切换触发两次请求。
      setCurrentThreadId(threadId);
      persistCurrentThreadId(threadId);
    },
    [persistCurrentThreadId],
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

  // 就绪且 agent 可用时，加载当前会话历史（初始化 + 切换共用这一条路径）。
  useEffect(() => {
    if (!ready || !currentThreadId || !agent) return;
    let cancelled = false;
    setLoadingThread(true);
    (async () => {
      try {
        const history = await getThreadMessages(currentThreadId);
        if (cancelled) return;
        agent.threadId = currentThreadId;
        agent.setMessages(history);
      } catch (err) {
        console.error("加载会话历史失败", err);
      } finally {
        if (!cancelled) setLoadingThread(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ready, currentThreadId, agent]);

  // 滚动到底部（瞬时，避免流式输出时 smooth 滚动卡顿）。
  const scrollToBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    isProgrammaticScrollRef.current = true;
    container.scrollTop = container.scrollHeight;
  }, []);

  // 用户手动滚动时记录是否已上滚；程序滚动产生的 scroll 事件则忽略。
  const handleScroll = useCallback(() => {
    if (isProgrammaticScrollRef.current) {
      isProgrammaticScrollRef.current = false;
      return;
    }
    const container = scrollContainerRef.current;
    if (!container) return;
    const { scrollTop, scrollHeight, clientHeight } = container;
    isUserScrollUpRef.current = scrollTop + clientHeight < scrollHeight;
  }, []);

  // 监听消息容器 DOM 变化：新消息（childList）与流式文本增长（characterData）
  // 都会触发，用户未上滚时自动滚到底部。characterData 是关键——它让
  // 「思考中…」在同一 DOM 节点内不断追加文字时也能跟随滚动。
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    container.addEventListener("scroll", handleScroll);
    const observer = new MutationObserver(() => {
      if (!isUserScrollUpRef.current) {
        scrollToBottom();
      }
    });
    observer.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    return () => {
      container.removeEventListener("scroll", handleScroll);
      observer.disconnect();
    };
  }, [handleScroll, scrollToBottom]);

  // 用户发送新消息时，重置「上滚」状态并强制滚到底部。
  const userMessageCount = (agent?.messages ?? []).filter(
    (m) => m.role === "user",
  ).length;
  useEffect(() => {
    isUserScrollUpRef.current = false;
    scrollToBottom();
  }, [userMessageCount, scrollToBottom]);

  const handleSelect = (threadId: string) => {
    switchThread(threadId);
  };

  const handleCreate = async () => {
    const created = await createThread();
    setThreads((prev) => [created, ...prev]);
    switchThread(created.thread_id);
  };

  const handleRename = async (threadId: string, title: string) => {
    const updated = await renameThread(threadId, title);
    setThreads((prev) =>
      prev.map((t) => (t.thread_id === threadId ? updated : t)),
    );
  };

  const handleDelete = async (threadId: string) => {
    // 1. 乐观移除：侧边栏立即消失，不等网络返回。
    const remaining = threads.filter((t) => t.thread_id !== threadId);
    setThreads(remaining);

    // 2. 若删除的是当前会话，立即切换到列表第一个（或新建），不阻塞 UI。
    if (threadId === currentThreadId) {
      if (remaining.length > 0) {
        switchThread(remaining[0].thread_id);
      } else {
        try {
          const created = await createThread();
          setThreads([created]);
          switchThread(created.thread_id);
        } catch (err) {
          console.error("新建会话失败", err);
        }
      }
    }

    // 3. 后台真正删除并刷新，同步服务端权威列表。
    try {
      await deleteThread(threadId);
      const list = await refreshThreads();
      setThreads(list);
    } catch (err) {
      console.error("删除会话失败", err);
      // 删除失败时回滚为服务端真实列表。
      const list = await refreshThreads().catch(() => threads);
      setThreads(list);
    }
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
      // 匿名 user_id 经 AG-UI forwarded_props 传给后端（clarifications Q6），
      // 后端 _extract_user_id 从 forwarded_props.user_id 读取（宪法 §5 runtime context）。
      await copilotkit.runAgent({
        agent,
        forwardedProps: { user_id: getOrCreateUserId() },
      });
    } catch (err) {
      console.error("发送失败", err);
    } finally {
      setSending(false);
    }
  };

  const loadKnowledgeMap = useCallback(async () => {
    setMapLoading(true);
    setMapError(null);
    try {
      const userId = getOrCreateUserId();
      const resp = await fetch(
        `/api/knowledge-map?user_id=${encodeURIComponent(userId)}`,
      );
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data: KnowledgeMapData = await resp.json();
      setMapData(data.goals ?? []);
    } catch (err) {
      console.error("加载知识版图失败", err);
      setMapError("加载知识版图失败，请稍后重试");
    } finally {
      setMapLoading(false);
    }
  }, []);

  const showMap = useCallback(() => {
    setView("map");
    void loadKnowledgeMap();
  }, [loadKnowledgeMap]);

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
        <header className="flex items-center justify-between border-b border-zinc-200 bg-white px-6 py-3">
          <div>
            <h1 className="text-lg font-semibold text-zinc-900">
              Cognia · AI 学习教练
            </h1>
            <p className="text-sm text-zinc-500">
              主动发现认知盲区，动态引导掌握知识点
            </p>
          </div>
          <button
            onClick={() => (view === "chat" ? showMap() : setView("chat"))}
            className="rounded-lg border border-zinc-300 px-3 py-1.5 text-sm text-zinc-700 transition hover:bg-zinc-100"
          >
            {view === "chat" ? "知识版图" : "返回对话"}
          </button>
        </header>

        <div
          ref={scrollContainerRef}
          className="min-h-0 flex-1 overflow-y-auto bg-white"
        >
          {view === "map" ? (
            <div className="p-6">
              {mapLoading ? (
                <div className="text-sm text-zinc-400">加载中…</div>
              ) : mapError ? (
                <div className="text-sm text-red-600">{mapError}</div>
              ) : mapData.length === 0 ? (
                <div className="py-16 text-center text-sm text-zinc-400">
                  还没有学习记录，去对话里开始一段学习吧
                </div>
              ) : (
                <KnowledgeMap goals={mapData} />
              )}
            </div>
          ) : !isReady || !ready || loadingThread ? (
            <div className="flex h-full items-center justify-center text-sm text-zinc-400">
              加载中…
            </div>
          ) : messages.length === 0 ? (
            <div className="flex h-full items-center justify-center text-sm text-zinc-400">
              开始一段新对话吧
            </div>
          ) : (
            displayMessages.map((item, idx) => {
              if (item.toolCall) {
                const rendered = renderToolCall({
                  toolCall: item.toolCall,
                  toolMessage: item.toolMessage,
                });
                if (rendered) {
                  return (
                    <div key={item.toolCall.id ?? `tool-${idx}`}>{rendered}</div>
                  );
                }
              }
              if (!item.message) {
                return (
                  <div key={`tool-${idx}`} className="px-4 py-1 text-xs text-zinc-400">
                    <span className="rounded bg-zinc-100 px-2 py-1">
                      🔧 {item.toolName || "工具"} 调用中…
                    </span>
                  </div>
                );
              }
              return (
                <MessageBubble
                  key={item.message.id ?? `msg-${idx}`}
                  message={item.message}
                  toolName={item.toolName}
                  isActiveReasoning={item.isActiveReasoning}
                />
              );
            })
          )}
        </div>

        {view === "chat" && (
          <div className="border-t border-zinc-200 bg-white p-4">
            <div className="flex gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
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
        )}
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" renderToolCalls={cogniaToolRenderers}>
      <ChatApp />
    </CopilotKit>
  );
}
