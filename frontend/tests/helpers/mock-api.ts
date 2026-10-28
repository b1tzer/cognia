import type { Page } from "@playwright/test";

// Cognia 前端所有浏览器端请求都走 /api/*（Next.js 代理到 FastAPI 后端）。
// UI 回归测试不依赖真实后端 / AI 模型 / PostgreSQL，用 page.route 拦截并返回
// 可控的 mock 数据。这样测试可离线、稳定、零外部成本地运行。

export type ThreadFixture = {
  thread_id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
};

export const THREAD_1: ThreadFixture = {
  thread_id: "thread-1",
  title: "测试会话",
  created_at: "2026-01-01T00:00:00.000Z",
  updated_at: "2026-01-02T00:00:00.000Z",
};

export const KNOWLEDGE_MAP = {
  user_id: "user-1",
  goals: [
    {
      goal: "Java 并发基础",
      points: [
        {
          id: "p1",
          name: "线程状态",
          description: "Java 线程的六种状态与流转",
          prerequisites: [],
        },
        {
          id: "p2",
          name: "锁机制",
          description: "synchronized 与 Lock",
          prerequisites: ["p1"],
        },
      ],
      proficiencies: { p1: "mastered", p2: "unassessed" },
    },
  ],
};

/**
 * 拦截所有 /api/** 请求并返回 mock 数据。
 * - /api/copilotkit：直接 abort，避免真实 Agent 连接 / SSE 流。
 * - /api/threads、/api/knowledge-map、/api/wiki、/api/feedback：按方法返回固定数据。
 */
export async function mockApi(page: Page): Promise<void> {
  // 隐藏 CopilotKit 开发期 Web Inspector（<cpk-web-inspector>）浮层。
  // 它是 dev-only 的调试浮层，会拦截页面按钮的 pointer 事件，导致 E2E 点击失败。
  await page.addInitScript(() => {
    // CSS 隐藏可能被 web-inspector 的 shadow DOM / 重挂载绕过，改用
    // MutationObserver 在元素出现时直接移除，彻底避免拦截 pointer 事件。
    const removeInspector = () => {
      document
        .querySelectorAll("cpk-web-inspector")
        .forEach((el) => el.remove());
    };
    const style = document.createElement("style");
    style.textContent =
      "cpk-web-inspector { display: none !important; pointer-events: none !important; }";
    (document.head || document.documentElement).appendChild(style);
    removeInspector();
    new MutationObserver(removeInspector).observe(document.documentElement, {
      childList: true,
      subtree: true,
    });
  });

  // 每个测试独立的进程内状态，保证「新建 / 重命名 / 删除」后列表能反映变更。
  let threads: ThreadFixture[] = [THREAD_1];
  let wikiPages: Array<Record<string, unknown>> = [];
  let seq = 0;

  const now = "2026-01-04T00:00:00.000Z";

  await page.route("**/api/**", async (route) => {
    const req = route.request();
    const url = new URL(req.url());
    const method = req.method();
    const path = url.pathname;

    // Agent runtime：UI 测试不跑真实模型。返回合法的 /info 结构（含空 agents），
    // 避免 CopilotKit 客户端抛错或触发 dev 错误浮层，从而挡住后续点击。
    if (path.startsWith("/api/copilotkit")) {      if (method === "GET" && path.endsWith("/info")) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            version: "1.71.2",
            agents: {
              default: { name: "default", description: "", className: "Z" },
            },
            audioFileTranscriptionEnabled: false,
            mode: "sse",
            threadEndpoints: {
              list: false,
              inspect: false,
              mutations: false,
              realtimeMetadata: false,
            },
            suggestions: false,
            a2uiEnabled: false,
            openGenerativeUIEnabled: false,
            telemetryDisabled: false,
          }),
        });
      }
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: "{}",
      });
    }

    // ---- threads ----
    if (path === "/api/threads" && method === "GET") {
      return route.fulfill({ json: threads });
    }
    if (path === "/api/threads" && method === "POST") {
      const created: ThreadFixture = {
        thread_id: `thread-new-${++seq}`,
        title: "新对话",
        created_at: now,
        updated_at: now,
      };
      threads = [created, ...threads];
      return route.fulfill({ json: created });
    }

    const threadMessages = path.match(/^\/api\/threads\/([^/]+)\/messages$/);
    if (threadMessages && method === "GET") {
      return route.fulfill({ json: { messages: [] } });
    }

    const threadById = path.match(/^\/api\/threads\/([^/]+)$/);
    if (threadById && method === "PATCH") {
const body = (req.postDataJSON() ?? {}) as {
        title?: string;
      };
      const id = threadById[1];
      const target = threads.find((t) => t.thread_id === id) ?? THREAD_1;
      const updated: ThreadFixture = {
        ...target,
        title: body.title ?? target.title,
      };
      threads = threads.map((t) => (t.thread_id === id ? updated : t));
      return route.fulfill({ json: updated });
    }
    if (threadById && method === "DELETE") {
      threads = threads.filter((t) => t.thread_id !== threadById[1]);
      return route.fulfill({ status: 200, body: "" });
    }

    // ---- feedback ----
    if (path === "/api/feedback" && method === "GET") {
      return route.fulfill({ json: {} });
    }

    // ---- knowledge-map ----
    if (path === "/api/knowledge-map" && method === "GET") {
      return route.fulfill({ json: KNOWLEDGE_MAP });
    }

    // ---- wiki ----
    if (path === "/api/wiki" && method === "GET") {
      return route.fulfill({ json: { pages: wikiPages } });
    }
    if (path === "/api/wiki" && method === "POST") {
const body = (req.postDataJSON() ?? {}) as {
        page_id?: string;
        title?: string;
        content_markdown?: string;
      };
      const pageId = body.page_id ?? `page-${++seq}`;
      const meta = {
        page_id: pageId,
        title: body.title ?? "未命名",
        path: `/${pageId}`,
        tags: [],
        parent_page_id: null,
        author: "user",
        source_thread_id: null,
        source_turns: [],
        evidence: [],
        updated_at: now,
        commit_hash: "abc1234",
      };
      wikiPages = [meta, ...wikiPages];
      return route.fulfill({ json: meta });
    }

    const wikiById = path.match(/^\/api\/wiki\/([^/]+)$/);
    if (wikiById && method === "GET") {
      return route.fulfill({
        json: {
          page_id: wikiById[1],
          title: "我的 Wiki",
          path: `/${wikiById[1]}`,
          tags: [],
          parent_page_id: null,
          author: "user",
          source_thread_id: null,
          source_turns: [],
          evidence: [],
          updated_at: now,
          commit_hash: "abc1234",
          content_markdown: "# 内容",
        },
      });
    }
    if (wikiById && method === "PUT") {
const body = (req.postDataJSON() ?? {}) as {
        title?: string;
      };
      const meta = {
        page_id: wikiById[1],
        title: body.title ?? "未命名",
        path: `/${wikiById[1]}`,
        tags: [],
        parent_page_id: null,
        author: "user",
        source_thread_id: null,
        source_turns: [],
        evidence: [],
        updated_at: now,
        commit_hash: "abc1234",
      };
      return route.fulfill({ json: meta });
    }
    if (wikiById && method === "DELETE") {
      return route.fulfill({ status: 200, body: "" });
    }

    // 未覆盖的 /api 请求统一 abort，避免意外打到真实后端。
    return route.abort();
  });
}
