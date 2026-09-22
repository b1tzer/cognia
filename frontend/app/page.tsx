"use client";

import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import WikiView from "./components/wiki-view";
import {
  cogniaToolRenderers,
  consumePendingPracticeAnswer,
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
import {
  deleteFeedback,
  listFeedback,
  upsertFeedback,
  type FeedbackMap,
} from "./lib/feedback";
import { getOrCreateUserId } from "./lib/user-id";

// localStorage 只保存「当前选中的 threadId」。会话列表与历史消息的权威数据
// 都在服务端 PostgreSQL；localStorage 不保存线程列表。
const THREAD_ID_KEY = "cognia:threadId";

const MessageBubble = memo(function MessageBubble({
  message,
  toolName,
  isActiveReasoning,
  feedback,
  onFeedback,
  showActions,
  onRetry,
}: {
  message: any;
  toolName?: string;
  isActiveReasoning?: boolean;
  feedback?: "up" | "down" | null;
  onFeedback?: (value: "up" | "down") => void;
  showActions?: boolean;
  onRetry?: () => void;
}) {
  const role = message.role as string;
  if (role === "tool") {
    return (
      <div className="px-4 py-1 text-xs text-muted">
        <span className="rounded bg-surface-muted px-2 py-1">
          🔧 {toolName || "工具"} 调用完成
        </span>
      </div>
    );
  }
  if (role === "reasoning") {
    const text = typeof message.content === "string" ? message.content : "";
    return (
      <div className="px-4 py-1">
        <details className="text-xs text-muted" open={isActiveReasoning}>
          <summary className="cursor-pointer select-none italic">
            {isActiveReasoning ? "思考中…" : "思考过程"}
          </summary>
          <div className="mt-1 whitespace-pre-wrap rounded-lg bg-surface-muted p-2 text-muted">
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
        <div className="max-w-[75%] whitespace-pre-wrap rounded-2xl bg-ink px-4 py-2 text-sm text-white">
          {text}
        </div>
      ) : (
        <div className="flex max-w-[85%] flex-col items-start gap-1">
          <div className="rounded-2xl bg-surface-muted px-4 py-2 text-sm text-foreground">
            <Markdown content={markdown} />
          </div>
          {showActions && (
            <div className="flex items-center gap-1 pl-1">
              <button
                onClick={() => onFeedback?.("up")}
                title="回答有帮助"
                className={`rounded-md px-2 py-1 text-sm transition ${
                  feedback === "up"
                    ? "bg-accent/15 text-accent"
                    : "text-muted hover:bg-surface-muted"
                }`}
              >
                👍
              </button>
              <button
                onClick={() => onFeedback?.("down")}
                title="回答有误"
                className={`rounded-md px-2 py-1 text-sm transition ${
                  feedback === "down"
                    ? "bg-accent/15 text-accent"
                    : "text-muted hover:bg-surface-muted"
                }`}
              >
                👎
              </button>
              <button
                onClick={onRetry}
                title="重新生成这条回答"
                className="rounded-md px-2 py-1 text-xs text-muted transition hover:bg-surface-muted hover:text-foreground"
              >
                ↻ 重新生成
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

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

  // 预计算：从下标 i 之后是否存在 assistant 正式回答。
  // 一次倒序遍历 O(n)，替代原先 reasoning 分支里 messages.slice(i+1).some() 的 O(n²)。
  const hasAnswerAfterFrom = new Array<boolean>(messages.length).fill(false);
  let seenAnswer = false;
  for (let i = messages.length - 1; i >= 0; i--) {
    hasAnswerAfterFrom[i] = seenAnswer;
    const x = messages[i];
    if (
      x.role === "assistant" &&
      typeof x.content === "string" &&
      x.content.trim()
    ) {
      seenAnswer = true;
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

    // 隐藏消息（如「练习作答记录」）只回传后端供 Agent 推理，不在此渲染。
    if (m.hidden) continue;

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
      display.push({
        message: m,
        isActiveReasoning: !hasAnswerAfterFrom[i] && isRunning,
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
  // 消息反馈（点赞 / 点踩）：{ messageId: "up" | "down" }，权威数据在服务端。
  const [feedbackMap, setFeedbackMap] = useState<FeedbackMap>({});
  // 视图切换：chat（对话）/ map（知识版图）/ wiki（个人 Wiki）
  const [view, setView] = useState<"chat" | "map" | "wiki">("chat");
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

  // 切换会话时拉取该会话的消息反馈，恢复点赞 / 点踩高亮。
  useEffect(() => {
    if (!ready || !currentThreadId) return;
    let cancelled = false;
    listFeedback(currentThreadId, getOrCreateUserId())
      .then((map) => {
        if (!cancelled) setFeedbackMap(map);
      })
      .catch(() => {
        // 拉取失败静默降级：feedback 高亮为空，不影响对话主流程。
      });
    return () => {
      cancelled = true;
    };
  }, [ready, currentThreadId]);

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
    // 流式输出时 characterData 高频触发，用 rAF 合并同一帧内的多次滚动请求。
    let rafId = 0;
    const observer = new MutationObserver(() => {
      if (isUserScrollUpRef.current) return;
      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        scrollToBottom();
      });
    });
    observer.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    return () => {
      container.removeEventListener("scroll", handleScroll);
      cancelAnimationFrame(rafId);
      observer.disconnect();
    };
  }, [handleScroll, scrollToBottom]);

  // 用户发送新消息时，重置「上滚」状态并强制滚到底部。
  const userMessageCount = useMemo(
    () => (agent?.messages ?? []).filter((m) => m.role === "user").length,
    [agent?.messages],
  );
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
      // 若上一道练习有作答结果，先作为一条隐藏的「练习作答记录」注入，
      // 让 Agent 在下一轮能读到学生选了哪个、对错如何（非阻塞方案）。
      const practice = consumePendingPracticeAnswer();
      if (practice) {
        const selectedOpt = practice.options[practice.selected_index] ?? "";
        const correctOpt =
          practice.correct_index != null
            ? (practice.options[practice.correct_index] ?? "")
            : "";
        const recordText =
          `【练习作答记录】题目：「${practice.question}」；` +
          `我选择了「${selectedOpt}」，` +
          (practice.is_correct
            ? "答对了。"
            : `答错了，正确答案是「${correctOpt}」。`);
        agent.addMessage({
          id: crypto.randomUUID(),
          role: "user",
          content: recordText,
          hidden: true,
        } as any);
      }

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

  // 中断当前正在生成的回答（CopilotKit v2 原生 abortRun）。
  const handleStop = useCallback(() => {
    agent?.abortRun();
  }, [agent]);

  // 重试：截断到最后一条 user 消息，重新生成回答。
  // 后端 ag_ui_langgraph 检测到「checkpoint 消息数 > 前端消息数」会自动
  // time-travel 到该 user 消息之前 fork 出新分支重新生成，无需额外回滚。
  const handleRetry = useCallback(async () => {
    if (!agent || sending) return;
    const msgs = agent.messages;
    let lastUserIndex = -1;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "user") {
        lastUserIndex = i;
        break;
      }
    }
    if (lastUserIndex < 0) return;
    // 保留最后一条 user 消息及其之前的历史，删除之后的 assistant / tool 回复。
    agent.setMessages(msgs.slice(0, lastUserIndex + 1));
    setSending(true);
    try {
      await copilotkit.runAgent({
        agent,
        forwardedProps: { user_id: getOrCreateUserId() },
      });
    } catch (err) {
      console.error("重新生成失败", err);
    } finally {
      setSending(false);
    }
  }, [agent, copilotkit, sending]);

  // 点赞 / 点踩：乐观更新 + 持久化；再点同一种取消，点另一种切换。
  const handleFeedback = useCallback(
    async (messageId: string, value: "up" | "down") => {
      if (!currentThreadId) return;
      const userId = getOrCreateUserId();
      const current = feedbackMap[messageId];
      if (current === value) {
        setFeedbackMap((prev) => {
          const next = { ...prev };
          delete next[messageId];
          return next;
        });
      } else {
        setFeedbackMap((prev) => ({ ...prev, [messageId]: value }));
      }
      try {
        if (current === value) {
          await deleteFeedback(currentThreadId, messageId, userId);
        } else {
          await upsertFeedback(currentThreadId, messageId, userId, value);
        }
      } catch (err) {
        console.error("反馈操作失败", err);
        // 失败回滚：重新拉取权威状态，保证与服务端一致。
        const map = await listFeedback(currentThreadId, userId).catch(
          () => feedbackMap,
        );
        setFeedbackMap(map);
      }
    },
    [currentThreadId, feedbackMap],
  );

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

  // 最后一条「正式回答」的 assistant 消息 id：只有它挂操作栏（重试 + 点赞点踩），
  // 中间的过渡思考 / 工具调用不提供这些操作。流式输出中暂不显示，等回答完整。
  const lastAssistantId = useMemo(() => {
    if (agent?.isRunning) return null;
    for (let i = displayMessages.length - 1; i >= 0; i--) {
      const m = displayMessages[i].message;
      if (
        m?.role === "assistant" &&
        typeof m.content === "string" &&
        m.content.trim()
      ) {
        return m.id;
      }
    }
    return null;
  }, [displayMessages, agent?.isRunning]);

  const renderMessageItem = (item: (typeof displayMessages)[number]) => {
    if (item.toolCall) {
      const rendered = renderToolCall({
        toolCall: item.toolCall,
        toolMessage: item.toolMessage,
      });
      if (rendered) {
        return <div className="px-4">{rendered}</div>;
      }
    }
    if (!item.message) {
      return (
        <div className="px-4 py-1 text-xs text-muted">
          <span className="rounded bg-surface-muted px-2 py-1">
            🔧 {item.toolName || "工具"} 调用中…
          </span>
        </div>
      );
    }
    const isLastAssistant =
      item.message?.role === "assistant" &&
      item.message?.id != null &&
      item.message.id === lastAssistantId;
    return (
      <MessageBubble
        message={item.message}
        toolName={item.toolName}
        isActiveReasoning={item.isActiveReasoning}
        feedback={
          isLastAssistant ? (feedbackMap[item.message.id] ?? null) : null
        }
        onFeedback={
          isLastAssistant
            ? (value) => void handleFeedback(item.message.id, value)
            : undefined
        }
        showActions={isLastAssistant}
        onRetry={isLastAssistant ? () => void handleRetry() : undefined}
      />
    );
  };

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
        <header className="flex items-center justify-between border-b border-line bg-surface px-6 py-3">
          <div>
            <h1 className="text-lg font-semibold text-foreground">
              Cognia · AI 学习教练
            </h1>
            <p className="text-sm text-muted">
              主动发现认知盲区，动态引导掌握知识点
            </p>
          </div>
          <div className="flex items-center gap-2">
            {(
              [
                { key: "chat", label: "对话" },
                { key: "map", label: "知识版图" },
                { key: "wiki", label: "Wiki" },
              ] as const
            ).map((tab) => (
              <button
                key={tab.key}
                onClick={() =>
                  tab.key === "map" ? showMap() : setView(tab.key)
                }
                className={`rounded-lg px-3 py-1.5 text-sm transition ${
                  view === tab.key
                    ? "bg-ink text-white"
                    : "text-foreground hover:bg-surface-muted"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </header>

        {view === "wiki" ? (
          <div className="min-h-0 flex-1 bg-surface">
            <WikiView currentThreadId={currentThreadId} />
          </div>
        ) : (
        <div
          ref={scrollContainerRef}
          className="min-h-0 flex-1 overflow-y-auto bg-surface"
        >
          {view === "map" ? (
            <div className="p-6">
              {mapLoading ? (
                <div className="text-sm text-muted">加载中…</div>
              ) : mapError ? (
                <div className="text-sm text-red-600">{mapError}</div>
              ) : mapData.length === 0 ? (
                <div className="py-16 text-center text-sm text-muted">
                  还没有学习记录，去对话里开始一段学习吧
                </div>
              ) : (
                <KnowledgeMap goals={mapData} />
              )}
            </div>
          ) : !isReady || !ready || loadingThread ? (
            <div className="flex h-full items-center justify-center text-sm text-muted">
              加载中…
            </div>
          ) : messages.length === 0 ? (
            <div className="flex h-full items-center justify-center text-sm text-muted">
              开始一段新对话吧
            </div>
          ) : (
            <div>
              {displayMessages.map((item, index) => {
                const key =
                  item.toolCall?.id ?? item.message?.id ?? `item-${index}`;
                return (
                  <div key={key}>
                    {renderMessageItem(item)}
                  </div>
                );
              })}
            </div>
          )}
        </div>
        )}

        {view === "chat" && (
          <div className="border-t border-line bg-surface p-4">
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
                className="flex-1 rounded-xl border border-line px-4 py-2 text-sm text-foreground outline-none focus:border-accent"
              />
              {sending ? (
                <button
                  onClick={handleStop}
                  className="rounded-xl bg-ink px-5 py-2 text-sm font-medium text-white transition hover:bg-ink/90"
                >
                  ⏹ 停止
                </button>
              ) : (
                <button
                  onClick={() => void handleSend()}
                  disabled={!input.trim()}
                  className="rounded-xl bg-ink px-5 py-2 text-sm font-medium text-white transition hover:bg-ink/90 disabled:opacity-40"
                >
                  发送
                </button>
              )}
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
