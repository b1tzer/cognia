import {
  CopilotRuntime,
  createCopilotRuntimeHandler,
  InMemoryAgentRunner,
} from "@copilotkit/runtime/v2";
import { LangGraphHttpAgent } from "@copilotkit/runtime/langgraph";

// 后端 AG-UI 端点（cognia/server.py，LangGraphAGUIAgent）
const AGENT_URL = process.env.AGENT_URL || "http://localhost:8123/";

const runtime = new CopilotRuntime({
  agents: {
    default: new LangGraphHttpAgent({
      url: AGENT_URL,
    }),
  },
  runner: new InMemoryAgentRunner(),
});

const handler = createCopilotRuntimeHandler({
  runtime,
  basePath: "/api/copilotkit",
});

export const GET = handler;
export const POST = handler;
export const PATCH = handler;
export const DELETE = handler;
