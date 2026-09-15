"use client";

import { useEffect, useState } from "react";
import { CopilotKit, CopilotChat } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";

// threadId 持久化 key：刷新页面（React remount）后必须读回同一个 threadId，
// 否则 CopilotKit 会重新 auto-mint 一个新 UUID，后端 checkpointer 按新
// thread_id 查不到历史，对话从零开始。
const THREAD_ID_KEY = "cognia:threadId";

function getOrCreateThreadId(): string {
  const existing = window.localStorage.getItem(THREAD_ID_KEY);
  if (existing) return existing;
  const id = crypto.randomUUID();
  window.localStorage.setItem(THREAD_ID_KEY, id);
  return id;
}

export default function Home() {
  const [threadId, setThreadId] = useState<string | null>(null);

  // 客户端挂载后再读 localStorage，避免 SSR 与客户端 hydration 不一致
  useEffect(() => {
    setThreadId(getOrCreateThreadId());
  }, []);

  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <div className="h-screen w-full flex flex-col">
        <header className="border-b bg-white px-6 py-3">
          <h1 className="text-lg font-semibold text-zinc-900">
            Cognia · AI 学习教练
          </h1>
          <p className="text-sm text-zinc-500">
            主动发现认知盲区，动态引导掌握知识点
          </p>
        </header>
        <div className="flex-1 min-h-0">
          {threadId ? (
            <CopilotChat
              threadId={threadId}
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
    </CopilotKit>
  );
}
