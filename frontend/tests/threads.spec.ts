import { test, expect } from "@playwright/test";
import { mockApi } from "./helpers/mock-api";

test.beforeEach(async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await expect(page.getByText("测试会话", { exact: true })).toBeVisible();
});

test("新建会话进入空态，不立即创建后端会话", async ({ page }) => {
  let createCalled = false;
  page.on("request", (req) => {
    if (req.method() === "POST" && req.url().includes("/api/threads")) {
      createCalled = true;
    }
  });

  await page.getByRole("button", { name: "新建会话" }).click();

  // 进入「新对话」空态，侧边栏列表不变、原会话仍在
  await expect(
    page.getByRole("heading", { name: "开始一段新对话吧" }),
  ).toBeVisible();
  await expect(page.getByText("测试会话", { exact: true })).toBeVisible();
  // 未触发后端创建
  expect(createCalled).toBe(false);
});

test("重复点击新建会话仍停留在空态，不产生多余会话", async ({ page }) => {
  let createCalled = false;
  page.on("request", (req) => {
    if (req.method() === "POST" && req.url().includes("/api/threads")) {
      createCalled = true;
    }
  });

  await page.getByRole("button", { name: "新建会话" }).click();
  await page.getByRole("button", { name: "新建会话" }).click();

  // 仍停留在「新对话」空态
  await expect(
    page.getByRole("heading", { name: "开始一段新对话吧" }),
  ).toBeVisible();
  // 未触发后端创建
  expect(createCalled).toBe(false);
});

test("发送第一条消息时才真正创建后端会话", async ({ page }) => {
  await page.getByRole("button", { name: "新建会话" }).click();

  const createReqPromise = page.waitForRequest(
    (req) => req.method() === "POST" && req.url().includes("/api/threads"),
  );

  await page
    .getByPlaceholder("输入消息，Enter 发送，Shift+Enter 换行")
    .fill("你好");
  await page.getByRole("button", { name: "发送" }).click();

  // 确认发送第一条消息后触发了后端创建会话
  await createReqPromise;
});

test("在 Wiki 页面点击侧边栏会话跳转回对话视图", async ({ page }) => {
  await page.getByRole("button", { name: "Wiki" }).click();
  await expect(page.getByText("暂无 wiki 页面", { exact: true })).toBeVisible();

  await page.getByText("测试会话", { exact: true }).click();
  await expect(
    page.getByPlaceholder("输入消息，Enter 发送，Shift+Enter 换行"),
  ).toBeVisible();
});

test("在知识版图页面点击侧边栏会话跳转回对话视图", async ({ page }) => {
  await page.getByRole("button", { name: "知识版图" }).click();
  await expect(
    page.getByRole("heading", { name: "Java 并发基础" }),
  ).toBeVisible();

  await page.getByText("测试会话", { exact: true }).click();
  await expect(
    page.getByPlaceholder("输入消息，Enter 发送，Shift+Enter 换行"),
  ).toBeVisible();
});

test("在 Wiki 页面新建会话跳转回对话视图", async ({ page }) => {
  await page.getByRole("button", { name: "Wiki" }).click();
  await expect(page.getByText("暂无 wiki 页面", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "新建会话" }).click();
  await expect(
    page.getByPlaceholder("输入消息，Enter 发送，Shift+Enter 换行"),
  ).toBeVisible();
  // 进入「新对话」空态
  await expect(
    page.getByRole("heading", { name: "开始一段新对话吧" }),
  ).toBeVisible();
});

test("重命名会话更新侧边栏标题", async ({ page }) => {
  // hover 到会话行文本，触发操作按钮显示（Tailwind .group-hover:flex）
  await page.getByText("测试会话", { exact: true }).hover();
  await page.getByRole("button", { name: "重命名" }).click();

  const input = page.getByPlaceholder("输入会话标题");
  await input.fill("改名后的会话");
  await input.press("Enter");

  await expect(page.getByText("改名后的会话", { exact: true })).toBeVisible();
});

test("删除会话需二次确认，可取消或确认删除", async ({ page }) => {
  // 先取消删除，会话仍在
  await page.getByText("测试会话", { exact: true }).hover();
  await page.getByRole("button", { name: "删除" }).click();
  await expect(page.getByRole("button", { name: "取消", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "取消", exact: true }).click();
  await expect(page.getByText("测试会话", { exact: true })).toBeVisible();

  // 再确认删除，会话从侧边栏移除
  await page.getByText("测试会话", { exact: true }).hover();
  await page.getByRole("button", { name: "删除" }).click();
  await page.getByRole("button", { name: "删除", exact: true }).click();
  await expect(page.getByText("测试会话", { exact: true })).toHaveCount(0);
});
