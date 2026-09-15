"use client";

import { CopilotKit, CopilotChat } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";

export default function Home() {
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
          <CopilotChat
            labels={{
              title: "Cognia 学习教练",
              initial:
                "你好！我是 Cognia，你的 AI 学习教练。\n请告诉我你想学什么（例如「Spring AOP」），我会先了解你的基础，再针对性教学。",
            }}
          />
        </div>
      </div>
    </CopilotKit>
  );
}
