import { defineConfig, devices } from "@playwright/test";

// Cognia 前端 UI 回归测试配置。
//
// 测试通过拦截浏览器端的 /api/** 请求返回 mock 数据（见 tests/helpers/mock-api.ts），
// 因此不依赖真实后端 / AI 模型 / PostgreSQL，也不产生任何外部调用。
// 只需本地 Next.js 服务提供静态页面与客户端 bundle。

const PORT = 3000;
const BASE_URL = `http://localhost:${PORT}`;

export default defineConfig({
  testDir: "./tests",
  // 每个测试用例超时（含首次 on-demand 编译，故放宽）。
  timeout: 60_000,
  // 测试相互独立，串行执行避免共享 dev server 下的状态干扰。
  fullyParallel: false,
  workers: 1,
  // 失败保留截图 + 首次重试保留 trace，用于定位问题。
  retries: process.env.CI ? 2 : 0,
  reporter: [
    ["list"],
    ["html", { open: "never" }],
  ],
  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  expect: {
    timeout: 15_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "pnpm dev",
    url: BASE_URL,
    reuseExistingServer: true,
    timeout: 120_000,
    env: {
      PORT: String(PORT),
    },
  },
});
